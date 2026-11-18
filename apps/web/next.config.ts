import type { NextConfig } from "next";

// The browser only talks to this app's origin. /api/* is forwarded to FastAPI,
// so auth cookies are first-party and no CORS is needed.
const apiUrl = (process.env.API_INTERNAL_URL ?? "http://localhost:8000").replace(/\/$/, "");

const nextConfig: NextConfig = {
  output: "standalone",
  poweredByHeader: false,
  // Ksarth streams replies; compression would buffer them.
  compress: false,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${apiUrl}/api/:path*` }];
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
        ],
      },
    ];
  },
};

export default nextConfig;
