import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end tests run against a real stack: MongoDB, the FastAPI app and the
 * Next.js server. Start them first (docker compose up, or see the README),
 * or set E2E_BASE_URL to point at a running deployment.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:3000",
    trace: "off",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "phone", use: { ...devices["Pixel 7"] } },
    // Many students use smaller Android phones; 360px is a common width in India.
    { name: "small-phone", use: { ...devices["Galaxy S9+"], viewport: { width: 360, height: 740 } } },
  ],
});
