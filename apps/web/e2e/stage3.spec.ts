import { expect, type Page, test } from "@playwright/test";

const PASSWORD = "a strong password 42";

async function expectNoSideScroll(page: Page) {
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow, `page is ${overflow}px wider than the screen`).toBeLessThanOrEqual(0);
}

async function signUp(page: Page, project: string) {
  await page.goto("/signup");
  await page.getByLabel("Your name").fill("Meera Iyer");
  await page.getByLabel("Email").fill(`e2e-s3-${project}-${Date.now()}-${Math.floor(Math.random() * 1e6)}@example.com`);
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

test("a student tries a job, reflects, and the reflection reaches the family report", async ({ page }, info) => {
  await signUp(page, info.project.name);

  await page.getByRole("link", { name: "Tryouts" }).click();
  await expect(page.getByRole("heading", { name: "Try the work before you choose it" })).toBeVisible();
  await expectNoSideScroll(page);
  await page.getByRole("link", { name: /The fractions mistake/ }).click();

  // Answers are hidden until submitting.
  await expect(page.getByText("Aarav's notebook")).toBeVisible();
  await expect(page.getByText("How a professional might write it")).toHaveCount(0);
  const submit = page.getByRole("button", { name: "Submit" });
  await expect(submit).toBeDisabled();

  await page.getByText("He adds the top numbers and the bottom numbers separately", { exact: false }).click();
  await page.getByText("Can you do it again more carefully?", { exact: false }).click();
  await page.getByLabel("Your answer").fill(
    "Good try Aarav. Imagine two rotis cut into halves and thirds; make the pieces the same size by cutting both into sixths. Then 3/6 + 2/6 = 5/6.",
  );
  await expect(page.getByText("3 of 3 answered")).toBeVisible();
  await expectNoSideScroll(page);
  await submit.click();

  // Objective tasks are marked exactly; with no AI key the written answer is honestly left ungraded.
  await expect(page.getByRole("heading", { name: /1 of 2 \(1 written answer not graded\)/ })).toBeVisible();
  await expect(page.getByText("Correct", { exact: true })).toBeVisible();
  await expect(page.getByText("Not quite", { exact: true })).toBeVisible();
  await expect(page.getByText("graded by the AI service, which isn't set up")).toBeVisible();
  await page.getByText("How a professional might write it").click();
  await expect(page.getByText("cut both rotis into sixths", { exact: false })).toBeVisible();

  await page.getByText("Quite a lot", { exact: true }).click();
  await page.getByText("Maybe", { exact: true }).click();
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.getByRole("button", { name: "Saved" })).toBeVisible();

  // The tryouts list remembers the attempt.
  await page.getByRole("link", { name: "Other tryouts" }).click();
  await expect(page.getByText(/Tried · 1 of 2 .*enjoyed it 4 of 5/)).toBeVisible();

  // Family report, in English and Hindi.
  await page.goto("/report");
  await expect(page.getByRole("heading", { name: "Career report for Meera" })).toBeVisible();
  await expect(page.getByText("Maybe could see themselves doing this").first()).toBeVisible();
  await expect(page.getByText("meera", { exact: false }).filter({ hasText: "@" })).toHaveCount(0);
  await page.getByRole("radio", { name: "हिन्दी" }).click();
  await expect(page.getByRole("heading", { name: "Meera की करियर रिपोर्ट" })).toBeVisible();
  await expect(page.locator("article[lang=hi]")).toBeVisible();
  await expectNoSideScroll(page);

  // A private link opens without signing in, and stops working once turned off.
  await page.getByRole("button", { name: "Create a Hindi link" }).click();
  const url = await page.locator("#share-url").inputValue();
  expect(url).toMatch(/\/r\/[\w-]{30,}$/);
  const visitor = await page.context().browser()!.newContext();
  const shared = await visitor.newPage();
  await shared.goto(url);
  await expect(shared.getByRole("heading", { name: "Meera की करियर रिपोर्ट" })).toBeVisible();
  await shared.getByRole("radio", { name: "English" }).click();
  await expect(shared.getByRole("heading", { name: "Career report for Meera" })).toBeVisible();
  await page.getByRole("button", { name: "Turn off link" }).click();
  await expect(page.getByRole("button", { name: "Create a Hindi link" })).toBeVisible();
  await shared.reload();
  await expect(shared.getByRole("heading", { name: "This link isn't working" })).toBeVisible();
  await visitor.close();
});

test("Ksarth points to the career guides behind its answer", async ({ page }, info) => {
  await signUp(page, info.project.name);
  await page.getByRole("link", { name: "Ksarth" }).click();
  await page.getByLabel("Message Ksarth").fill("How do I become a chartered accountant after commerce?");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("From Pathify's career guides")).toBeVisible();
  await expectNoSideScroll(page);
  await page.getByRole("link", { name: "[1] Chartered accountant" }).click();
  await expect(page).toHaveURL(/\/careers\/chartered-accountant$/);
});
