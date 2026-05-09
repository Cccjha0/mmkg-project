import { expect, test } from "@playwright/test";

test("core demo pages load and KG remains responsive at neighbours=2", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Model Performance" })).toBeVisible();
  await expect(page.getByText("Performance Summary")).toBeVisible();

  await page.getByRole("button", { name: "Attribute Completion" }).click();
  await expect(page.getByRole("heading", { name: "Attribute Completion" })).toBeVisible();
  await expect(page.getByText("Entity Information")).toBeVisible();

  await page.getByRole("button", { name: "Knowledge Graph" }).click();
  await expect(page.getByRole("heading", { name: "Knowledge Graph Explorer" })).toBeVisible();
  await expect(page.getByLabel("Search")).toHaveValue("Pants");
  await expect(page.getByText("Graph Statistics")).toBeVisible();

  await page.getByLabel("Prune").fill("3");

  const searchResponse = page.waitForResponse(
    (response) =>
      response.url().includes("http://127.0.0.1:5000/search/1/2/3/Pants") &&
      response.status() === 200,
    { timeout: 20_000 },
  );
  await page.getByLabel("Neighbours").fill("2");

  const response = await searchResponse;
  const graph = await response.json();
  expect(graph.nodes.length).toBeLessThanOrEqual(251);
  expect(graph.links.length).toBeLessThanOrEqual(500);

  await expect(page.getByLabel("Neighbours")).toHaveValue("2");
  await expect(page.getByRole("heading", { name: "Knowledge Graph Explorer" })).toBeVisible();
  await expect(page.getByText("Total Graph Nodes")).toBeVisible();
});
