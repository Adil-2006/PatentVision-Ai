/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // In local dev, proxy /api/* to local FastAPI backend (http://127.0.0.1:8000).
    // In production, proxy to NEXT_PUBLIC_API_URL or API_PROXY_TARGET (or deployed Render backend)
    // so same-origin /api/* calls never return 404.
    const defaultTarget =
      process.env.NODE_ENV === "development"
        ? "http://127.0.0.1:8000"
        : "https://patentvision-ai-8.onrender.com";
    const target = (
      process.env.NEXT_PUBLIC_API_URL ||
      process.env.API_PROXY_TARGET ||
      defaultTarget
    ).replace(/\/+$/, "");

    return [{ source: "/api/:path*", destination: `${target}/:path*` }];
  },
};

export default nextConfig;
