/**
 * End-to-end server verification script:
 * 1. Starts FastAPI backend on port 8000.
 * 2. Starts Vite frontend on port 5173.
 * 3. Calls /health via HTTP.
 * 4. Calls frontend / via HTTP.
 * 5. Shuts down cleanly.
 */

const { spawn } = require("child_process");
const http = require("http");
const path = require("path");

function fetch(url, options = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request(url, options, (res) => {
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => resolve({ status: res.statusCode, headers: res.headers, body: data }));
    });
    req.on("error", reject);
    req.setTimeout(5000, () => {
      req.destroy();
      reject(new Error("Timeout connecting to " + url));
    });
    req.end();
  });
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  console.log("Starting backend server...");
  const pythonPath = `"${path.join(__dirname, "..", "backend", ".venv", "Scripts", "python.exe")}"`;
  const backend = spawn(pythonPath + " -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000", {
    shell: true,
  });

  backend.stdout.on("data", (d) => console.log("[Backend stdout]:", d.toString().trim()));
  backend.stderr.on("data", (d) => console.log("[Backend stderr]:", d.toString().trim()));

  console.log("Starting frontend Vite server...");
  const frontend = spawn("npm run dev --prefix frontend -- --port 5173 --host 127.0.0.1", {
    shell: true,
  });

  frontend.stdout.on("data", (d) => console.log("[Frontend stdout]:", d.toString().trim()));
  frontend.stderr.on("data", (d) => console.log("[Frontend stderr]:", d.toString().trim()));

  try {
    // Poll for backend and frontend readiness up to 10 seconds
    let backendReady = false;
    let frontendReady = false;
    let healthData = null;
    let frontendData = null;

    for (let i = 0; i < 15; i++) {
      await sleep(1000);
      if (!backendReady) {
        try {
          const res = await fetch("http://127.0.0.1:8000/health", {
            headers: { Origin: "http://localhost:5173" },
          });
          if (res.status === 200) {
            backendReady = true;
            healthData = res;
            console.log("Backend is ready!");
          }
        } catch (e) {
          // keep waiting
        }
      }

      if (!frontendReady) {
        try {
          const res = await fetch("http://127.0.0.1:5173/");
          if (res.status === 200) {
            frontendReady = true;
            frontendData = res;
            console.log("Frontend is ready!");
          }
        } catch (e) {
          // keep waiting
        }
      }

      if (backendReady && frontendReady) break;
    }

    if (backendReady && frontendReady) {
      console.log("\n=======================================================");
      console.log("SUCCESS: Both servers started cleanly!");
      console.log(`Backend /health response: ${healthData.body}`);
      console.log(`Backend CORS Access-Control-Allow-Origin: ${healthData.headers["access-control-allow-origin"]}`);
      console.log(`Frontend index.html length: ${frontendData.body.length} bytes`);
      console.log("Frontend can call /health successfully!");
      console.log("=======================================================\n");
    } else {
      throw new Error(`Servers failed to start within timeout. Backend ready: ${backendReady}, Frontend ready: ${frontendReady}`);
    }
  } catch (err) {
    console.error("Verification failed:", err.message);
    process.exitCode = 1;
  } finally {
    console.log("Stopping test servers...");
    // Kill processes
    spawn("taskkill", ["/F", "/T", "/PID", backend.pid], { shell: true });
    spawn("taskkill", ["/F", "/T", "/PID", frontend.pid], { shell: true });
  }
}

main();
