# Phase 1 accuracy review

Manual review of the enrichment agent on 10 test leads.

- **Run:** `20261108T072343Z`, model `gemini-3.5-flash-lite`, code at commit `d44a5a3`
- **Reviewed:** 113 rows, one per lead per field, with one row per item for `location`, `tech_stack` and `recent_news`
- **Blank verdicts skipped:** 0
- **Scoring:** `correct` and `ok_null` both count as right

## Results

| Outcome | Count | Share |
|---|---|---|
| Right (87 correct + 2 ok_null) | 89 | 78.8% |
| Wrong | 2 | 1.8% |
| Missing | 22 | 19.5% |

Without the 10 rows for the Gmail lead with no hint, where there is nothing to look up, the score
is 89 right of 103 (86.4%), 2 wrong (1.9%) and 12 missing (11.7%).

Of the 89 rows that had a value, 85 are marked correct. By confidence: high 52 of 52, medium 7 of 7,
low 26 of 30.

### By field

| Field | Reviewed | Right | Wrong | Missing |
|---|---|---|---|---|
| domain | 10 | 9 (90%) | 0 | 1 (10%) |
| industry | 10 | 9 (90%) | 0 | 1 (10%) |
| employee_range | 10 | 8 (80%) | 0 | 2 (20%) |
| location | 11 | 10 (91%) | 0 | 1 (9%) |
| ownership | 10 | 7 (70%) | 1 (10%) | 2 (20%) |
| funding | 10 | 3 (30%) | 0 | 7 (70%) |
| website_summary | 10 | 9 (90%) | 0 | 1 (10%) |
| tech_stack | 18 | 16 (89%) | 1 (6%) | 1 (6%) |
| recent_news | 14 | 9 (64%) | 0 | 5 (36%) |
| company_name | 10 | 9 (90%) | 0 | 1 (10%) |

`funding` looks worst, but only 1 of its 7 missing rows may be a real gap (see below).

## Wrong rows

| Lead | Field | Value | Likely cause |
|---|---|---|---|
| info@daraz.pk | ownership | `acquired_by: Alibaba Group Holding Limited` | **Unresolved, not a sign of an invented value.** The collected data supports it: LinkedIn lists Alibaba as parent company, and CB Insights and Dealroom show an acquisition in May 2018. The same sources also call Daraz "Privately Held", so this may be a question of how to label a subsidiary. The model also cited a low-trust Tavily page when LinkedIn said the same thing. Needs the reviewer's reason. |
| info@daraz.pk | tech_stack | (empty) | **Detector gap in our code.** The HTML was fetched (431 KB, 16 script files), but the scripts come mostly from `g.lazcdn.com`, Alibaba/Lazada's own platform, and none of the 19 signatures match. |

## Missing rows

The pipeline does not save what it collects, so the original data no longer exists. To answer this,
the same searches and scrapes were repeated on 2026-11-08 without the LLM. Search results can change
between runs, so this is a close approximation of what the model saw.

| Lead | Field | In the collected data? | Why it is empty or flagged |
|---|---|---|---|
| info@daraz.pk | funding | Yes | $55.6M Series A (2015) was found. Cleared by the rule for acquired companies |
| info@kyliecosmetics.com | funding | Yes | $600M Coty majority deal, an acquisition rather than a round. Cleared by rule |
| info@microsoft.com | funding | Yes | Junk figures on a profile page (2.86B TWD). Cleared by the rule for public companies |
| info@unilever.com | funding | Yes | Grant figures on a profile page (20.90M USD). Cleared by the rule for public companies |
| info@systemsltd.com | recent_news | Yes | An Oct 17, 2026 item on Egypt investment was collected, from a page about its regional unit. Own-site items seen in an earlier run had no date, and undated news is dropped |
| info@soapplai.com | employee_range | Yes, value returned | "Headcount: 5" gave `1-10`. Excel displayed that as `10-Jan` |
| info@kyliecosmetics.com | ownership | Yes, value returned | `acquired_by: Coty` was returned. Wikipedia gives Coty 51% and Kylie Jenner 44.1%, so the value may be seen as incomplete |
| info@systemsltd.com | funding | Searched, nothing relevant | No private round found. The company is public |
| info@soapplai.com | funding | Never fetched | Searches returned other companies. Founded 2026 and still on a waitlist, so there may be nothing to find |
| info@soapplai.com | recent_news | Never fetched | Same as above |
| info@allbirds.com | recent_news | Never fetched | Only older articles and policy pages. No news-specific search exists |
| info@kyliecosmetics.com | recent_news | Never fetched | No dated company news in the results |
| test.lead9@gmail.com | company_name, domain, website_summary, industry, employee_range, ownership, funding, location, tech_stack, recent_news (10 rows) | Never fetched, by design | Free email with no company hint: the agent correctly stops |

### Breakdown

| Group | Rows |
|---|---|
| Free email with no hint, by design | 10 |
| Found, then cleared by a funding rule | 4 |
| Value was returned but flagged | 2 |
| In the data, not returned | 1 |
| Not in the data | 5 |
| **Total** | **22** |

## Top 3 fixes, by impact

1. **Add a dedicated news search.** `recent_news` is the weakest field with real gaps: 4 of the 12
   missing rows outside the Gmail lead. Today news comes only from a combined funding and news
   query. A separate search limited to the last 12 months should fill most of these and give the
   date check something reliable to work with. This field also feeds the personalised first message.
2. **Drop search results that aren't about the company before the LLM sees them.** For Soapplai,
   most search results were about unrelated companies, including funding news for other startups.
   The model handled this correctly here, but small companies are the likely real-world leads, and
   this is the most probable source of future wrong values. Keep a result only if it mentions the
   company name or domain.
3. **Extend tech detection.** Add signatures for the platforms that are currently missed, starting
   with what Daraz loads. This is the one clear code error in the review, and it is low effort, but
   affects few rows, so it ranks third.

## Notes for the next review

- **Rows that lower the score without being errors:** 16 of the 22 missing rows are expected nulls
  (the Gmail lead and rule-cleared funding) or flagged values. An `n/a` verdict would separate these.
- **One verdict to re-check:** Gymshark `location` (New York) is marked correct at high confidence.
  Its page lists Solihull and New York without naming a headquarters, and Wikipedia gives Solihull.
  If this is wrong, the score is 88 right and 3 wrong, and headquarters selection becomes a candidate fix.
- **Excel:** it turns `1-10` into a date and may re-save the file as tab-separated text.
- **Sample size:** 10 leads and one reviewer, so treat these percentages as indicative.
