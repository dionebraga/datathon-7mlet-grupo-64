/** @type {import('next').NextConfig} */
// Proxy /api/* to the FastAPI backend so the browser never hits CORS and the
// Python API stays untouched. Override the backend URL with API_URL.
const API = process.env.API_URL || "http://localhost:8000";

const nextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/:path*` }];
  },
};

export default nextConfig;
