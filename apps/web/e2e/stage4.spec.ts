import { expect, type Page, test } from "@playwright/test";

const PASSWORD = "a strong password 42";

async function expectNoSideScroll(page: Page) {
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow, `page is ${overflow}px wider than the screen`).toBeLessThanOrEqual(0);
}

async function signUp(page: Page, project: string, name = "Kabir Shah") {
  await page.goto("/signup");
  await page.getByLabel("Your name").fill(name);
  await page.getByLabel("Email").fill(`e2e-s4-${project}-${Date.now()}-${Math.floor(Math.random() * 1e6)}@example.com`);
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Create account" }).click();
  await expect(page).toHaveURL(/\/onboarding$/);
  await page.getByText("Class 9 or 10", { exact: true }).click();
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByText("Class 10", { exact: true }).click();
  await page.getByText("A parent or guardian knows I'm using Pathify").click();
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("button", { name: "Finish setup" }).click();
  await expect(page).toHaveURL(/\/home$/);
}

test("a student makes a plan, ticks a step and earns XP", async ({ page }, info) => {
  await signUp(page, info.project.name);
  await expect(page.getByRole("heading", { name: "Level 1" })).toBeVisible();
  await expect(page.getByText("Confirm your email")).toBeVisible();

  await page.goto("/plan");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Pick a route and make it yours");
  await page.getByLabel("Any career in the library").or(page.getByLabel("Career", { exact: true })).selectOption("lawyer");
  await page.getByRole("button", { name: /Five-year integrated law degree|law/i }).first().click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("lawyer");
  await expectNoSideScroll(page);

  const steps = page.getByRole("list", { name: "Steps on your route" });
  await expect(steps).toContainText("CLAT");
  await page.getByRole("button", { name: /^Tick off: / }).first().click();
  await expect(page.getByText(/^1 of \d+ done$/)).toBeVisible();

  await page.getByLabel("Add your own step").fill("Read one judgment a week");
  await page.getByRole("button", { name: "Add", exact: true }).click();
  await expect(steps).toContainText("Read one judgment a week");

  await page.goto("/home");
  // 20 for setting up, 25 for the plan, 15 for the step.
  await expect(page.getByText("60", { exact: true })).toBeVisible();
  await expect(page.getByText("Route planner (earned)")).toBeAttached();
});

test("forgot password never reveals whether an account exists", async ({ page }) => {
  await page.goto("/signin");
  await page.getByRole("link", { name: "Forgot your password?" }).click();
  await expect(page).toHaveURL(/\/forgot-password$/);
  await page.getByLabel("Email").fill("nobody-at-all@example.com");
  await page.getByRole("button", { name: "Send me a link" }).click();
  await expect(page.getByText("If an account uses that email, we've sent a link")).toBeVisible();

  await page.goto("/reset-password?token=not-a-real-token-at-all-xyz");
  await page.getByLabel("New password").fill("another good password 9");
  await page.getByRole("button", { name: "Save and sign in" }).click();
  await expect(page.getByText("This link has expired or was already used.")).toBeVisible();
});

test("a counsellor sees a student who joined with the code", async ({ browser }, info) => {
  const counsellor = await browser.newPage();
  await signUp(counsellor, `${info.project.name}-c`, "Meena Iyer");
  await counsellor.goto("/counsellor");
  await counsellor.getByLabel("New class").fill("Class 10 B");
  await counsellor.getByLabel("School").fill("Kendriya Vidyalaya");
  await counsellor.getByRole("button", { name: "Create class" }).click();
  const code = (await counsellor.getByText(/^[A-Z2-9]{6}$/).textContent())!.trim();

  const student = await browser.newPage();
  await signUp(student, `${info.project.name}-s`, "Kabir Shah");
  await student.goto("/settings");
  await student.getByLabel("Join your school's class").fill(code);
  await student.getByText("I agree to share a summary").click();
  await student.getByRole("button", { name: "Join class" }).click();
  await expect(student.getByText("Class 10 B, Kendriya Vidyalaya")).toBeVisible();

  await counsellor.reload();
  await expect(counsellor.getByRole("cell", { name: /Kabir Shah/ })).toBeVisible();
  await expect(counsellor.getByText("Hasn't taken the interest assessment")).toBeVisible();
  await expectNoSideScroll(counsellor);
});
