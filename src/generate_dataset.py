"""
Generate a realistic synthetic LoanTap dataset for loan default prediction.
This script creates a dataset matching the LoanTap data dictionary with
realistic distributions and correlations between features.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

N = 10000  # Number of samples

# ─── Loan Amount ───
loan_amnt = np.random.lognormal(mean=9.5, sigma=0.6, size=N).astype(int)
loan_amnt = np.clip(loan_amnt, 1000, 40000)

# ─── Term ───
term = np.random.choice([' 36 months', ' 60 months'], size=N, p=[0.72, 0.28])

# ─── Grade & Sub Grade ───
grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
grade_probs = [0.18, 0.28, 0.22, 0.15, 0.10, 0.05, 0.02]
grade = np.random.choice(grades, size=N, p=grade_probs)

sub_grades = []
for g in grade:
    sub_num = np.random.choice([1, 2, 3, 4, 5])
    sub_grades.append(f"{g}{sub_num}")
sub_grade = np.array(sub_grades)

# ─── Interest Rate (correlated with grade) ───
grade_rate_map = {'A': 7.0, 'B': 10.5, 'C': 14.0, 'D': 17.5, 'E': 21.0, 'F': 24.5, 'G': 27.0}
int_rate = np.array([grade_rate_map[g] + np.random.normal(0, 1.5) for g in grade])
int_rate = np.clip(int_rate, 5.32, 30.99)
int_rate = np.round(int_rate, 2)

# ─── Installment (derived from loan amount, term, and rate) ───
term_months = np.array([36 if '36' in t else 60 for t in term])
monthly_rate = int_rate / 100 / 12
installment = loan_amnt * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
installment = np.round(installment, 2)

# ─── Employment Title ───
emp_titles = [
    'Teacher', 'Manager', 'Registered Nurse', 'RN', 'Supervisor', 'Sales',
    'Owner', 'Driver', 'Office Manager', 'General Manager', 'Director',
    'Analyst', 'Engineer', 'Software Engineer', 'Account Manager',
    'Project Manager', 'Operations Manager', 'Technician', 'Accountant',
    'Administrative Assistant', 'Vice President', 'Mechanic', 'Nurse',
    'Police Officer', 'Attorney', 'Consultant', 'Server', 'Electrician',
    'Team Lead', 'Data Analyst', 'Marketing Manager', 'Physician',
    'Branch Manager', 'Warehouse Manager', 'Program Manager', 'IT Manager',
    'Financial Analyst', 'Cashier', 'Assistant Manager', 'Coordinator'
]
emp_title = np.random.choice(emp_titles, size=N)

# ─── Employment Length ───
emp_lengths = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
               '6 years', '7 years', '8 years', '9 years', '10+ years']
emp_length_probs = [0.05, 0.06, 0.08, 0.08, 0.07, 0.08, 0.07, 0.06, 0.06, 0.05, 0.34]
emp_length = np.random.choice(emp_lengths, size=N, p=emp_length_probs)

# ─── Home Ownership ───
home_ownership = np.random.choice(['RENT', 'MORTGAGE', 'OWN', 'OTHER'],
                                   size=N, p=[0.40, 0.44, 0.12, 0.04])

# ─── Annual Income ───
annual_inc = np.random.lognormal(mean=11.0, sigma=0.65, size=N).astype(int)
annual_inc = np.clip(annual_inc, 12000, 500000)

# ─── Verification Status ───
verification_status = np.random.choice(['Verified', 'Source Verified', 'Not Verified'],
                                        size=N, p=[0.33, 0.35, 0.32])

# ─── Issue Date ───
start_date = datetime(2015, 1, 1)
end_date = datetime(2020, 12, 31)
date_range = (end_date - start_date).days
issue_d = [(start_date + timedelta(days=random.randint(0, date_range))).strftime('%b-%Y') for _ in range(N)]

# ─── Purpose ───
purposes = ['debt_consolidation', 'credit_card', 'home_improvement', 'other',
            'major_purchase', 'medical', 'small_business', 'car', 'vacation',
            'moving', 'house', 'wedding', 'educational', 'renewable_energy']
purpose_probs = [0.35, 0.22, 0.12, 0.10, 0.06, 0.04, 0.03, 0.03, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005]
purpose = np.random.choice(purposes, size=N, p=purpose_probs)

# ─── Title ───
title_map = {
    'debt_consolidation': 'Debt consolidation',
    'credit_card': 'Credit card refinancing',
    'home_improvement': 'Home improvement',
    'other': 'Other',
    'major_purchase': 'Major purchase',
    'medical': 'Medical expenses',
    'small_business': 'Business',
    'car': 'Car financing',
    'vacation': 'Vacation',
    'moving': 'Moving and relocation',
    'house': 'Home buying',
    'wedding': 'Wedding',
    'educational': 'Educational',
    'renewable_energy': 'Green loan'
}
title = np.array([title_map.get(p, 'Other') for p in purpose])

# ─── DTI ───
dti = np.random.lognormal(mean=2.8, sigma=0.45, size=N)
dti = np.clip(dti, 0, 50)
dti = np.round(dti, 2)

# ─── Earliest Credit Line ───
earliest_cr_line = [(start_date - timedelta(days=random.randint(365*3, 365*35))).strftime('%b-%Y') for _ in range(N)]

# ─── Open Accounts ───
open_acc = np.random.poisson(lam=11, size=N)
open_acc = np.clip(open_acc, 1, 40)

# ─── Public Records ───
pub_rec = np.random.choice([0, 1, 2, 3, 4], size=N, p=[0.78, 0.15, 0.04, 0.02, 0.01])

# ─── Revolving Balance ───
revol_bal = np.random.lognormal(mean=9.2, sigma=1.2, size=N).astype(int)
revol_bal = np.clip(revol_bal, 0, 200000)

# ─── Revolving Utilization ───
revol_util = np.random.beta(2, 3, size=N) * 100
revol_util = np.round(revol_util, 1)

# ─── Total Accounts ───
total_acc = open_acc + np.random.poisson(lam=12, size=N)
total_acc = np.clip(total_acc, 2, 80)

# ─── Initial List Status ───
initial_list_status = np.random.choice(['w', 'f'], size=N, p=[0.60, 0.40])

# ─── Application Type ───
application_type = np.random.choice(['Individual', 'Joint App'], size=N, p=[0.88, 0.12])

# ─── Mortgage Accounts ───
mort_acc = np.random.choice(range(0, 12), size=N,
                             p=[0.30, 0.25, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005])

# ─── Public Record Bankruptcies ───
pub_rec_bankruptcies = np.random.choice([0, 1, 2, 3], size=N, p=[0.85, 0.11, 0.03, 0.01])

# ─── Address ───
states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI',
          'NJ', 'VA', 'WA', 'AZ', 'MA', 'IN', 'TN', 'MO', 'MD', 'WI',
          'CO', 'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT', 'NV']
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
          'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
          'Fort Worth', 'Columbus', 'Charlotte', 'Indianapolis', 'San Francisco',
          'Seattle', 'Denver', 'Washington', 'Nashville', 'Oklahoma City', 'El Paso',
          'Boston', 'Portland', 'Las Vegas', 'Memphis', 'Louisville', 'Baltimore', 'Milwaukee']
zip_codes = [str(random.randint(10000, 99999)) for _ in range(N)]
address = [f"{random.randint(100,9999)} {random.choice(['Main St', 'Oak Ave', 'Elm St', 'Park Rd', 'Cedar Ln', 'Maple Dr', 'Pine St', 'Lake Rd', 'Hill Ave', 'River Rd'])}\n{random.choice(cities)}, {random.choice(states)} {z}" for z in zip_codes]

# ─── Loan Status (Target Variable) ───
# Create realistic default probability based on features
grade_default_prob = {'A': 0.05, 'B': 0.12, 'C': 0.20, 'D': 0.28, 'E': 0.35, 'F': 0.42, 'G': 0.50}
base_prob = np.array([grade_default_prob[g] for g in grade])

# Adjust by DTI
dti_factor = (dti - dti.mean()) / dti.std() * 0.05
# Adjust by income (higher income = lower default)
inc_factor = -(np.log(annual_inc) - np.log(annual_inc).mean()) / np.log(annual_inc).std() * 0.03
# Adjust by term
term_factor = np.array([0.05 if '60' in t else 0 for t in term])

default_prob = np.clip(base_prob + dti_factor + inc_factor + term_factor, 0.01, 0.95)
loan_status_binary = np.random.binomial(1, default_prob)
loan_status = np.where(loan_status_binary == 0, 'Fully Paid', 'Charged Off')

# ─── Create DataFrame ───
df = pd.DataFrame({
    'loan_amnt': loan_amnt,
    'term': term,
    'int_rate': int_rate,
    'installment': installment,
    'grade': grade,
    'sub_grade': sub_grade,
    'emp_title': emp_title,
    'emp_length': emp_length,
    'home_ownership': home_ownership,
    'annual_inc': annual_inc,
    'verification_status': verification_status,
    'issue_d': issue_d,
    'loan_status': loan_status,
    'purpose': purpose,
    'title': title,
    'dti': dti,
    'earliest_cr_line': earliest_cr_line,
    'open_acc': open_acc,
    'pub_rec': pub_rec,
    'revol_bal': revol_bal,
    'revol_util': revol_util,
    'total_acc': total_acc,
    'initial_list_status': initial_list_status,
    'application_type': application_type,
    'mort_acc': mort_acc,
    'pub_rec_bankruptcies': pub_rec_bankruptcies,
    'address': address
})

# Introduce some missing values realistically
missing_cols = {
    'emp_title': 0.04,
    'emp_length': 0.03,
    'revol_util': 0.005,
    'mort_acc': 0.08,
    'pub_rec_bankruptcies': 0.005,
    'dti': 0.002,
    'title': 0.01
}

for col, frac in missing_cols.items():
    mask = np.random.choice([True, False], size=N, p=[frac, 1-frac])
    df.loc[mask, col] = np.nan

# Save
os.makedirs('data', exist_ok=True)
df.to_csv('data/LoanTapData.csv', index=False)
print(f"Dataset generated: {df.shape}")
print(f"\nLoan Status Distribution:")
print(df['loan_status'].value_counts(normalize=True).round(4) * 100)
print(f"\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\nSaved to data/LoanTapData.csv")
