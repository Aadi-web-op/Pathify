import { expect, type Page, test } from "@playwright/test";

const PASSWORD = "a strong password 42";

// Next.js renders its own empty role="alert" route announcer; skip it.
const appAlert = (page: Page) => page.locator('[role="alert"]:not(#__next-route-announcer__)');

/** Pages must fit the screen: anything wider makes phones zoom out and misplace taps. */
async function expectNoSideScroll(page: Page) {
  // clientWidth stays at the device width; innerWidth grows when a phone zooms out to fit wide content.
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow, `page is ${overflow}px wider than the screen`).toBeLessThanOrEqual(0);
}

function uniqueEmail(project: string) {
  return `e2e-${project}-${Date.now()}-${Math.floor(Math.random() * 1e6)}@example.com`;
}

async function signUp(page: Page, email: string, name = "Asha Verma") {
  await page.goto("/signup");
  await page.getByLabel("Your name").fill(name);
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Create account" }).click();
  await expect(page).toHaveURL(/\/onboarding$/);
}

async function onboardClass11(page: Page) {
  await expect(page.getByRole("heading", { name: "Where are you right now?" })).toBeVisible();
  await expectNoSideScroll(page);
  await page.getByText("Class 11 or 12", { exact: true }).click();
  await page.getByRole("button", { name: "Continue" }).click();

  await expect(page.getByRole("heading", { name: "Tell us about school" })).toBeVisible();
  await page.getByText("Class 12", { exact: true }).click();
  await page.getByLabel("Board").selectOption("cbse");
  const next = page.getByRole("button", { name: "Continue" });
  await expect(next).toBeDisabled();
  await page.getByText("Science (PCM)", { exact: true }).click();
  await page.getByText("A parent or guardian knows I'm using Pathify").click();
  await next.click();

  await expect(page.getByRole("heading", { name: "What do you want help with?" })).toBeVisible();
  // The stage's usual goal is pre-selected; add one more.
  await expect(page.getByRole("checkbox", { name: /Pick exams and a degree/ })).toBeChecked();
  await page.getByText("Just explore", { exact: true }).click();
  await page.getByRole("button", { name: "Continue" }).click();

  await expect(page.getByRole("heading", { name: "A little about you" })).toBeVisible();
  await page.getByLabel("Things you enjoy").fill("Robotics");
  await page.getByLabel("Things you enjoy").press("Enter");
  await page.getByRole("button", { name: "+ Drawing" }).click();
  await page.getByLabel("State or union territory").selectOption("Maharashtra");
  await page.getByRole("button", { name: "Finish setup" }).click();
  await expect(page).toHaveURL(/\/home$/);
}

test("a new student signs up, sets up a profile, talks to Ksarth and deletes the account", async ({ page }, info) => {
  const email = uniqueEmail(info.project.name);

  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Don't guess your future. Map it.");
  await expectNoSideScroll(page);

  await signUp(page, email);
  await onboardClass11(page);

  // Home reflects the saved profile.
  await expect(page.getByRole("heading", { name: "Hi Asha, here's your road ahead" })).toBeVisible();
  await expect(page.getByText("Class 12, Science (PCM)").first()).toBeVisible();
  const route = page.getByRole("list", { name: "Decisions ahead of you" });
  await expect(route.getByRole("listitem").first()).toHaveAttribute("aria-current", "step");
  await expect(route).toContainText("Exams and degree");
  await expect(page.getByText("Robotics, Drawing")).toBeVisible();
  await expectNoSideScroll(page);

  // A suggested question goes straight to Ksarth and gets a streamed reply.
  await page.getByRole("link", { name: /Which entrance exams fit my stream/ }).click();
  await expect(page).toHaveURL(/\/ksarth$/);
  await expect(page.getByRole("status").filter({ hasText: "Ksarth is in offline mode" })).toBeVisible();
  await expect(page.getByText(/You asked: .Which entrance exams fit my stream/)).toBeVisible();
  await expectNoSideScroll(page);

  // Model output is rendered as text, never as HTML.
  await page.getByLabel("Message Ksarth").fill('<img src=x onerror="window.__xss=1"> hello');
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText('<img src=x onerror="window.__xss=1"> hello').first()).toBeVisible();
  await expect(page.locator("section[aria-label=Conversation] img")).toHaveCount(0);
  expect(await page.evaluate(() => (window as unknown as { __xss?: number }).__xss)).toBeUndefined();

  // History survives a reload.
  await page.reload();
  await expect(page.getByText("Which entrance exams fit my stream?").first()).toBeVisible();

  // Theme choice applies immediately and is saved.
  await page.getByRole("link", { name: "Settings" }).click();
  await page.getByText("Dark", { exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expectNoSideScroll(page);

  // Sign out, then back in: returning students land on Home.
  await page.getByRole("button", { name: "Sign out" }).last().click();
  await expect(page).toHaveURL(/\/signin$/);
  await page.goto("/home");
  await expect(page).toHaveURL(/\/signin\?next=%2Fhome$/);
  await page.getByLabel("Email").fill(email.toUpperCase());
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/home$/);

  // Delete the account.
  await page.getByRole("link", { name: "Settings" }).click();
  const deleteButton = page.getByRole("button", { name: "Delete my account" });
  await expect(deleteButton).toBeDisabled();
  await page.getByLabel(`Type ${email} to confirm`).fill(email);
  await deleteButton.click();
  await expect(page).toHaveURL(/\/$/);
  await page.goto("/signin");
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(appAlert(page)).toContainText("don't match");
});

test("wrong password shows a clear error and the form keeps its values", async ({ page }) => {
  await page.goto("/signin");
  await page.getByLabel("Email").fill("nobody@example.com");
  await page.getByLabel("Password").fill("not the password");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(appAlert(page)).toHaveText("That email and password don't match. Check them and try again.");
  await expect(page.getByLabel("Email")).toHaveValue("nobody@example.com");
});

test("an expired access token is refreshed without the student noticing", async ({ page, context }, info) => {
  await signUp(page, uniqueEmail(info.project.name), "Ravi Kumar");
  await onboardClass11(page);
  await context.clearCookies({ name: "pathify_access" });
  await page.reload();
  await expect(page.getByRole("heading", { name: "Hi Ravi, here's your road ahead" })).toBeVisible();
});
