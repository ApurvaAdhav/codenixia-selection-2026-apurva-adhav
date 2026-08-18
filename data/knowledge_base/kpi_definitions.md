# KPI Definitions

## Revenue
Total sales value generated before costs are deducted. Calculated as
Quantity x Unit Price, summed across all transactions in the period.

## Cost
The total cost of goods sold (COGS) for the units sold, including
production/procurement cost. Does not include marketing or overhead.

## Profit
Revenue minus Cost for the same period. This is gross profit, not net
profit (it does not subtract overhead, salaries, or marketing spend).

## Profit Margin
Profit divided by Revenue, expressed as a percentage. A healthy margin for
this business is considered to be above 25%. Margins below 15% should be
flagged for review.

## Significant Sales Drop
A decline in revenue of 10% or more compared to the prior period of equal
length (e.g., this 30 days vs the previous 30 days) is classified as a
"significant drop" and should trigger an investigation per the Anomaly
Response Policy.

## Anomaly
A data point (typically a single day's revenue for a Region+Product
combination) that falls well outside the normal statistical range for that
series, as detected by the Isolation Forest model. Anomalies can be
"above normal" (positive surprises, e.g. a promotion) or "below normal"
(negative surprises, e.g. stockouts, quality issues, competitor action).

## Top Contributor
When explaining a change in overall revenue, the "top contributor" is the
Region, Product, or Segment with the largest absolute dollar change
(positive or negative) versus the prior period. Contributors are ranked by
absolute dollar impact, not percentage, because a 50% drop in a tiny
segment matters less than a 10% drop in a large one.

## Customer Segment
Sales are grouped into three segments: Consumer (individual retail buyers),
Corporate (large business accounts), and Small Business (SMB accounts).
Corporate accounts typically have lower per-unit margins but higher volume.
