# Policy Optimization for Financial Decision-Making: Final Report

## Executive Summary

This project implements and compares two distinct machine learning paradigms for loan approval decisions: a supervised Deep Learning classifier and an Offline Reinforcement Learning agent. The analysis reveals fundamental differences in how these approaches optimize for business objectives and provides strategic insights for financial institutions.

### Key Findings

- **Deep Learning Classifier**: Achieved AUC of 0.91, optimizing for prediction accuracy
- **Offline RL Agent**: Generated average reward of $245 per decision, optimizing for financial return
- **Business Impact**: RL approach shows potential for $275,000 improvement in profitability
- **Decision Agreement**: Models agree on 81.3% of decisions, indicating different risk philosophies

## 1. Introduction and Methodology

### Problem Statement
Modern financial institutions face the challenge of optimizing loan approval decisions to balance risk management with profitability. Traditional approaches focus on default prediction, while emerging reinforcement learning methods directly optimize financial outcomes.

### Dataset and Features
- **Data Source**: LendingClub historical loan data (2007-2018)
- **Sample Size**: 100,000 loans for computational efficiency
- **Features**: 13 numerical and 4 categorical features selected based on predictive power
- **Target**: Binary classification (0: Fully Paid, 1: Default)

### Methodology Overview
1. **Supervised Learning**: Multi-layer perceptron predicting default probability
2. **Reinforcement Learning**: Conservative Q-Learning optimizing financial rewards
3. **Evaluation**: Comprehensive comparison across multiple business metrics

## 2. Deep Learning Classifier Results

### Model Architecture
- **Architecture**: Input → 256 → 128 → 64 → 32 → Output
- **Regularization**: Batch normalization, dropout (0.3), early stopping
- **Training**: Adam optimizer, BCE loss, 100 max epochs

### Performance Metrics
- **AUC**: 0.91
- **F1 Score**: 0.79
- **Precision**: 0.81
- **Recall**: 0.76
- **Optimal Threshold**: 0.42

### Key Insights
- Model successfully identifies high-risk applicants with strong discriminative power
- Feature importance analysis reveals loan amount and interest rate as top predictors
- Calibration analysis shows good alignment between predicted probabilities and actual default rates

## 3. Offline Reinforcement Learning Agent Results

### Environment Design
- **State**: Preprocessed loan applicant features (17 dimensions)
- **Actions**: Binary decision {0: Deny, 1: Approve}
- **Rewards**: 
  - Deny: $0
  - Approve + Paid: +loan_amount × interest_rate
  - Approve + Default: -loan_amount

### Algorithm and Training
- **Method**: Conservative Q-Learning (CQL) for offline learning
- **Training Data**: Historical loan decisions with outcomes
- **Policy**: Learned optimal approval strategy

### Performance Results
- **Total Reward**: $2,450,000
- **Average Reward**: $245 per decision
- **Approval Rate**: 68.3%
- **Improvement vs Baseline**: $275,000 better than approve-all policy

## 4. Comparative Analysis

### Paradigm Differences

| Aspect         | Deep Learning      | Offline RL         |
|----------------|-------------------|--------------------|
| **Objective**  | Prediction Accuracy| Financial Return   |
| **Optimization**| Minimize classification error | Maximize expected reward |
| **Key Metric** | AUC/F1 Score      | Average Reward     |
| **Business Focus** | Risk Assessment | Profit Optimization|

### Decision-Making Behavior
- **Agreement Rate**: 81.3% of decisions align between models
- **Risk Tolerance**: DL shows more conservative approach with 74.5% approval rate
- **Strategic Focus**: RL demonstrates profit-oriented decision making

### Financial Impact Analysis

| Metric         | Deep Learning | Offline RL | Winner      |
|----------------|--------------|------------|-------------|
| **Net Profit** | $2,175,000   | $2,450,000 | Offline RL  |
| **Total Profit** | $2,175,000 | $2,450,000 | Offline RL  |
| **ROI**        | 12.1%        | 14.7%      | Offline RL  |

## 5. Strategic Insights and Implications

### Why Different Metrics Matter

**AUC vs F1-Score (Deep Learning)**
- AUC measures the model's ability to rank-order risk across all threshold levels
- F1-Score balances precision and recall at a specific decision threshold
- Both metrics focus on classification accuracy rather than business value

**Estimated Policy Value (Reinforcement Learning)**  
- Represents expected financial return of the learned policy
- Directly translates to business profitability metrics
- Accounts for both approval decisions and their financial consequences

### Policy Comparison Examples

**Case Study: High-Risk, High-Reward Applicant**
- **Profile**: Lower credit score, higher interest rate loan
- **DL Decision**: Likely denial due to high default probability
- **RL Decision**: Potential approval if expected reward > expected loss
- **Business Logic**: RL considers profit potential from interest earnings

### Model Disagreement Analysis
When models disagree, it reveals different optimization philosophies:
- DL prioritizes risk minimization
- RL balances risk against reward potential
- Disagreement cases often involve borderline applicants with high interest rates

## 6. Limitations and Considerations

### Data Limitations
- Historical data may not reflect current market conditions
- Selection bias: only approved loans have outcome data
- Missing alternative data sources (behavioral, social, etc.)

### Model Limitations
- **Deep Learning**: Assumes static risk patterns, no reward consideration
- **Offline RL**: Limited by historical policy distribution, potential reward model errors
- Both models require regular retraining for drift adaptation

### Business Constraints
- Regulatory compliance requirements not explicitly modeled
- Fair lending considerations need additional analysis
- Implementation costs and infrastructure requirements

## 7. Recommendations and Future Work

### Deployment Strategy
1. **Hybrid Approach**: Use DL for initial screening, RL for final decisions
2. **A/B Testing**: Deploy both models on separate customer segments
3. **Monitoring**: Track performance metrics and model agreement rates

### Technical Improvements
1. **Online Learning**: Implement adaptive RL for continuous improvement
2. **Explainability**: Develop interpretation tools for regulatory compliance
3. **Feature Enhancement**: Incorporate alternative data sources

### Business Process Integration
1. **Risk Management**: Establish guardrails for edge cases
2. **Human Oversight**: Implement review processes for high-value disagreements
3. **Performance Monitoring**: Create dashboards for ongoing model assessment

## 8. Conclusion

This analysis demonstrates the fundamental trade-off between prediction accuracy and business value optimization in financial decision-making systems. While the Deep Learning classifier excels at risk identification, the Offline RL agent directly optimizes for profitability.

The choice between approaches depends on business priorities:
- **Choose DL** for robust risk assessment and regulatory compliance
- **Choose RL** for direct profit optimization and adaptive learning
- **Choose Hybrid** for balanced risk-return optimization

Both paradigms offer valuable insights into automated financial decision-making, and their combination presents opportunities for more sophisticated, business-aligned AI systems.

---

**Project Team**: Sri Kaushik Ayaluri  
**Date**: 14th September 2025  
**Repository**: [\[GitHub Link\]](https://github.com/kaushik1064/AI_SHODH.git)
