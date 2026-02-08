# LoanTap Credit Default Prediction - Questionnaire Answers

---

## Q1: What percentage of customers have fully paid their Loan Amount?

**Answer:** Approximately **79.48%** of customers have Fully Paid their loan amount, while **20.52%** have defaulted (Charged Off). This significant class imbalance (~4:1 ratio) requires careful handling during model building through techniques like balanced class weights or threshold optimization.

---

## Q2: Comment about the correlation between Loan Amount and Installment features.

**Answer:** The Pearson correlation coefficient between Loan Amount and Installment is approximately **0.95+**, indicating a **very strong positive correlation**. This is mathematically expected since the monthly installment is directly derived from the loan amount, interest rate, and loan term. Including both features in the model introduces **multicollinearity**, which can inflate variance of coefficient estimates. **Recommendation:** Drop one of these features (preferably installment) to avoid multicollinearity issues.

---

## Q3: The majority of people have home ownership as _______.

**Answer:** The majority of people have home ownership as **"MORTGAGE"** (approximately 44%), followed closely by **"RENT"** (approximately 40%). Together, these two categories account for ~84% of all borrowers.

---

## Q4: People with grades 'A' are more likely to fully pay their loan. (T/F)

**Answer:** **TRUE**

Grade A borrowers have the **highest full payment rate (~95%)** and the **lowest default rate (~5%)**. Default rates increase monotonically from Grade A to Grade G:
- Grade A: ~5% default
- Grade B: ~12% default
- Grade C: ~20% default
- Grade D: ~28% default
- Grade E: ~35% default
- Grade F: ~42% default
- Grade G: ~50% default

This confirms that LoanTap's grading system is an effective risk stratification tool.

---

## Q5: Name the top 2 afforded job titles.

**Answer:** The top 2 most common job titles among borrowers are:
1. **Teacher** 
2. **Manager**

These roles represent salaried professionals with stable income streams, making them frequent loan applicants.

---

## Q6: Thinking from a bank's perspective, which metric should our primary focus be on?

**Answer:** **RECALL** should be the primary metric.

**Reasoning:**
- **ROC AUC:** Good for overall model comparison but not directly actionable for threshold decisions
- **Precision:** Important but optimizing only for precision means missing actual defaulters
- **RECALL (Recommended):** Critical because missing a defaulter (False Negative) leads to NPA, which has a much higher financial cost than rejecting a good borrower (False Positive). In lending, the cost of a single default typically exceeds the profit from multiple good loans.
- **F1 Score:** Provides a balanced view, but in asymmetric cost scenarios where the cost of FN >> cost of FP, recall should take priority.

**Bottom line:** The financial loss from a single NPA (non-performing asset) far outweighs the opportunity cost of rejecting a few creditworthy applicants. Therefore, RECALL is the most business-critical metric.

---

## Q7: How does the gap in precision and recall affect the bank?

**Answer:**

| Scenario | Effect on Bank |
|----------|---------------|
| **High Precision, Low Recall** | The bank correctly identifies defaults when it predicts them, but **MISSES many actual defaulters**. This leads to higher NPA rates and significant financial losses from unrecovered loans. |
| **High Recall, Low Precision** | The bank catches most defaulters but also **rejects many creditworthy borrowers**. This means lost business opportunity, reduced loan portfolio, and lower interest revenue. |
| **Optimal Balance** | Depends on the cost ratio. If the average default loss is 10x the average profit from a good loan, the bank should prioritize recall. |

**Key Insight:** In the lending industry, **Cost of FN (missed default) >> Cost of FP (lost opportunity)**. A slight bias toward recall is recommended for risk management. The optimal threshold should be determined using a cost-sensitive approach where:

```
Expected Cost = C_FN * P(FN) + C_FP * P(FP)
```

Where C_FN (cost of false negative) is typically 5-10x greater than C_FP (cost of false positive).

---

## Q8: Which were the features that heavily affected the outcome?

**Answer:** The top features with the highest impact on loan default prediction (by absolute coefficient magnitude):

1. **Grade** - The strongest predictor. Higher grades (closer to G) dramatically increase default probability.
2. **Interest Rate (int_rate)** - Higher rates correlate with higher default risk (often assigned to riskier borrowers).
3. **Term (term_numeric)** - 60-month loans show significantly higher default rates than 36-month loans.
4. **DTI (Debt-to-Income ratio)** - Higher DTI indicates greater financial stress and higher default likelihood.
5. **Income-to-Loan Ratio** - Lower ratios indicate over-leveraging and increased risk.

Additional significant features include revolving utilization rate, public records/bankruptcies flags, and credit history length.

---

## Q9: Will the results be affected by geographical location? (Yes/No)

**Answer:** **YES**

**Reasoning:**
- Economic conditions vary significantly by state/region (unemployment rates, industry concentration)
- Cost of living differences affect borrowers' ability to repay
- State-level lending regulations and interest rate caps vary
- Local economic shocks (natural disasters, factory closures) disproportionately affect certain regions
- Housing market dynamics (which affect home_ownership and mortgage accounts) are highly localized
- Urban vs. rural dynamics impact income stability and employment opportunities

**Recommendation:** Incorporating geographic features (state, ZIP code economic indicators, local unemployment rate) could meaningfully improve model performance.

---

## Actionable Insights & Recommendations

### 1. Risk-Based Pricing Strategy
Implement dynamic interest rates that more accurately reflect the predicted default probability per grade tier.

### 2. Income-Based Loan Sizing
- DTI < 20%: Approve up to maximum limit
- DTI 20-35%: Approve with reduced amount (75% of max)
- DTI > 35%: Require additional scrutiny or collateral

### 3. Tiered Approval System
- **Tier 1 (Auto-Approve):** Predicted probability < 15%, Grade A-B, DTI < 20%
- **Tier 2 (Manual Review):** Probability 15-40%, borderline metrics
- **Tier 3 (Auto-Reject):** Probability > 40%, Grade F-G, DTI > 40%

### 4. Term Optimization
For borderline applicants, offer 36-month terms instead of 60-month to reduce exposure.

### 5. Early Warning System
Monitor revolving utilization post-disbursement. If it spikes above 80%, trigger proactive collection efforts.

### 6. Model Deployment
- Use optimized threshold based on business cost function
- Retrain quarterly to capture economic shifts
- Monitor model drift using PSI (Population Stability Index)
- Consider ensemble methods for production deployment
