import { expect, type Page, test } from "@playwright/test";

import itemBank from "../../../data/assessment.json";
import catalog from "../../../data/careers.json";

const PASSWORD = "a strong password 42";
const DIMENSION_OF = new Map(itemBank.items.map((i) => [i.text, i.dimension]));

async function expectNoSideScroll(page: Page) {
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow, `page is ${overflow}px wider than the screen`).toBeLessThanOrEqual(0);
}

async function signUpAsClass11Commerce(page: Page, project: string) {
  await page.goto("/signup");
  await page.getByLabel("Your name").fill("Riya Sharma");
  await page.getByLabel("Email").fill(`e2e-assess-${project}-${Date.now()}-${Math.floor(Math.random() * 1e6)}@example.com`);
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Create account" }).click();
  await expect(page).toHaveURL(/\/onboarding$/);
  await page.getByText("Class 11 or 12", { exact: true }).click();
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByText("Commerce", { exact: true }).click();
  await page.getByText("A parent or guardian knows I'm using Pathify").click();
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("button", { name: "Finish setup" }).click();
  await expect(page).toHaveURL(/\/home$/);
}

/** Answer like a student who loves investigating and helping, and dislikes organising. */
const ANSWER: Record<string, string> = {
  R: "Dislike",
  I: "Like a lot",
  A: "Not sure",
  S: "Like a lot",
  E: "Dislike",
  C: "Dislike a lot",
};

async function answerQuestions(page: Page, count: number) {
  for (let i = 0; i < count; i++) {
    const statement = page.locator("#statement");
    const text = await statement.innerText();
    const dimension = DIMENSION_OF.get(text);
    expect(dimension, `unknown statement: ${text}`).toBeTruthy();
    await page.getByRole("button", { name: ANSWER[dimension!], exact: true }).click();
    if (i < count - 1) await expect(statement).not.toHaveText(text);
  }
}

test("a Class 11 Commerce student takes the assessment and gets an honest career map", async ({ page }, info) => {
  await signUpAsClass11Commerce(page, info.project.name);

  await page.getByRole("link", { name: "Take the interest assessment" }).click();
  await expect(page.getByRole("heading", { name: "What do you enjoy doing?" })).toBeVisible();
  await page.getByRole("button", { name: "Start" }).click();
  await expect(page.getByText("1 of 30")).toBeVisible();
  await expectNoSideScroll(page);

  // Back undoes an answer and returns to the same statement.
  const first = await page.locator("#statement").innerText();
  await page.getByRole("button", { name: "Like", exact: true }).click();
  await expect(page.getByText("2 of 30")).toBeVisible();
  await page.getByRole("button", { name: "Back" }).click();
  await expect(page.locator("#statement")).toHaveText(first);

  // Leaving and coming back resumes where the student was.
  await answerQuestions(page, 10);
  await page.getByRole("link", { name: "Save and finish later" }).click();
  await expect(page.getByRole("link", { name: "Continue (10 of 30)" })).toBeVisible();
  await page.getByRole("link", { name: "Continue (10 of 30)" }).click();
  await expect(page.getByText("11 of 30")).toBeVisible();
  await answerQuestions(page, 20);

  await expect(page).toHaveURL(/\/results$/);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Investigative");
  await expect(page.getByRole("heading", { name: "Your career map" })).toBeVisible();
  await expect(page.getByRole("img", { name: /Routes from where you are to your top matches/ })).toBeVisible();
  await expectNoSideScroll(page);

  // Every top match explains itself and shows a route.
  const cards = page.locator("article[id^=match-]");
  await expect(cards).toHaveCount(3);
  for (const card of await cards.all()) {
    await expect(card.getByRole("heading", { name: "Why it fits" })).toBeVisible();
    await expect(card.getByRole("list", { name: /Steps to become/ })).toBeVisible();
  }

  // Medicine needs Biology, so a Commerce student is told it isn't open from their stream.
  await expect(page.getByText("not open from your stream").first()).toBeVisible();

  // Career guides are public and linked from the results.
  await cards.first().getByRole("link", { name: "Full career guide" }).click();
  await expect(page).toHaveURL(/\/careers\/[a-z-]+$/);
  await expect(page.getByRole("heading", { name: "Routes in India" })).toBeVisible();
  await expectNoSideScroll(page);

  // Home now summarises the matches.
  await page.goto("/home");
  await expect(page.getByRole("heading", { name: "Your top matches" })).toBeVisible();
});

test("the career library is public and filters by stream", async ({ page }) => {
  await page.goto("/careers");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Careers in India, route by route");
  const count = page.getByRole("status").filter({ hasText: "careers" });
  const total = catalog.careers.length;
  await expect(count).toContainText(`${total} of ${total}`);

  await page.getByText("Right after Class 10", { exact: true }).click();
  await expect(page.getByRole("link", { name: "Electrician" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Doctor", exact: true })).toHaveCount(0);

  await page.getByText("Any stream", { exact: true }).click();
  await page.getByLabel("Search careers").fill("doctor");
  await page.getByRole("link", { name: "Doctor", exact: true }).click();
  await expect(page).toHaveURL(/\/careers\/doctor$/);
  await expect(page.getByText("NEET-UG", { exact: true })).toBeVisible();
  await expect(page).toHaveTitle(/How to become a doctor in India/);
  await expectNoSideScroll(page);
});
