import assert from "node:assert/strict";
import test from "node:test";
import {
  buildFallbackDailyReport,
  generateDailyReport,
  type ArticleInput,
} from "./pipeline";

function article(
  category: ArticleInput["category"],
  index: number,
  summary?: string,
): ArticleInput {
  return {
    sourceId: `${category}-source-${index % 2}`,
    source: `Source ${index % 2}`,
    title: `${category} title ${index}`,
    url: `https://example.com/${category}/${index}`,
    excerpt: `${category} excerpt ${index}`,
    summary,
    category,
    publishedAt: new Date(Date.now() - index * 1_000),
  };
}

test("buildFallbackDailyReport preserves enriched summaries and category caps", () => {
  const articles = [
    ...Array.from({ length: 7 }, (_, i) =>
      article("tech", i, i === 0 ? "enriched tech summary" : undefined),
    ),
    ...Array.from({ length: 6 }, (_, i) => article("finance", i)),
    ...Array.from({ length: 4 }, (_, i) => article("politics", i)),
  ];

  const report = buildFallbackDailyReport(articles);

  assert.equal(report.tech_briefs.length, 5);
  assert.equal(report.finance_briefs.length, 5);
  assert.equal(report.politics_briefs.length, 3);
  assert.equal(report.tech_briefs[0]?.summary, "enriched tech summary");
  assert.equal(report.finance_briefs[0]?.summary, "finance excerpt 0");
});

test("generateDailyReport falls back after repeated truncated JSON responses", async () => {
  const articles = [article("tech", 0, "summary")];
  let attempts = 0;

  const { report } = await generateDailyReport(
    articles,
    async () => {
      attempts += 1;
      throw new SyntaxError("Unexpected end of JSON input");
    },
    [0, 0, 0],
  );

  assert.equal(attempts, 4);
  assert.equal(report.tech_briefs.length, 1);
  assert.equal(report.tech_briefs[0]?.summary, "summary");
});

test("generateDailyReport rethrows account-level auth failures immediately", async () => {
  const articles = [article("tech", 0)];
  let attempts = 0;

  await assert.rejects(
    generateDailyReport(
      articles,
      async () => {
        attempts += 1;
        throw Object.assign(new Error("401 invalid API key"), { status: 401 });
      },
      [0, 0, 0],
    ),
    /401 invalid API key/,
  );
  assert.equal(attempts, 1);
});

test("generateDailyReport degrades to fallback when the relay keeps 400ing", async () => {
  const articles = [article("tech", 0, "enriched summary")];
  let attempts = 0;

  const { report } = await generateDailyReport(
    articles,
    async () => {
      attempts += 1;
      // Shape mirrors the OpenAI SDK's APIError: status carried on the error.
      throw Object.assign(
        new Error(
          "400 The request could not be processed. Please check the request parameters.",
        ),
        { status: 400 },
      );
    },
    [0, 0, 0],
  );

  assert.equal(attempts, 4);
  assert.equal(report.tech_briefs.length, 1);
  assert.equal(report.tech_briefs[0]?.summary, "enriched summary");
});

test("generateDailyReport succeeds after transient provider 400s", async () => {
  const articles = [article("tech", 0)];
  let attempts = 0;

  const { report } = await generateDailyReport(
    articles,
    async () => {
      attempts += 1;
      if (attempts < 3) {
        throw Object.assign(new Error("400 bad channel"), { status: 400 });
      }
      return {
        hero_headline: "h",
        daily_overview: "o",
        tech_briefs: [
          {
            title: "t",
            url: "https://example.com/x",
            source: "s",
            summary: "s",
            importance: 5,
          },
        ],
        finance_briefs: [],
        politics_briefs: [],
        editor_note: "n",
        keywords: ["k"],
      };
    },
    [0, 0, 0],
  );

  assert.equal(attempts, 3);
  assert.equal(report.tech_briefs.length, 1);
});
