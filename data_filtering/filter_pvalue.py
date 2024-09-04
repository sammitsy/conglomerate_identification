import numpy as np
import scipy.stats as stats

#Hypothesis 1: Proportion of keyword occurrence (higher in non-conglomerates)
# Observed counts for each keyword in conglomerates and non-conglomerates
observed_counts = {
    "Enantiomerically Pure": [86, 321],
    "Optically Pure": [32, 176],
    "Resolution": [526, 1476],
    "Desymmetrization": [16, 156],
    "Desymmetrisation": [1, 4],
    "Resolved": [85, 145],
    "Non-racemic": [9, 29],
    "Scalemic": [22, 16],
    "Enantioselective": [302, 1140],
    "Enantioselectivity": [131, 586],
}

# Total counts involved in the test for conglomerates and non-conglomerates
total_congloms = 1703
total_non_congloms = 4519

# Calculate p-values for keyword occurrence
p_values_keywords = {}
for keyword, counts in observed_counts.items():
    # Observed proportions
    prop_congloms = counts[0] / total_congloms
    prop_non_congloms = counts[1] / total_non_congloms
    
    # Pooled proportion
    pooled_prop = (counts[0] + counts[1]) / (total_congloms + total_non_congloms)
    
    # Standard error
    std_error = np.sqrt(pooled_prop * (1 - pooled_prop) * (1/total_congloms + 1/total_non_congloms))
    
    # Z statistic
    z_stat = (prop_non_congloms - prop_congloms) / std_error
    
    # One-tailed p-value (since the hypothesis is that non-congloms have a higher occurrence)
    p_value = 1 - stats.norm.cdf(z_stat)
    
    # Store the p-value
    p_values_keywords[keyword] = p_value

# Hypothesis 2: Proportion of Chiral Structures (higher in conglomerates)
# Observed counts for chiral structures
chiral_counts = {
    "Chiral": [(1608+299), (15287+ 3699)],
    "Achiral": [(95+41), (3448+1365)],
    "Not Chiral & Achiral": [(32+3), (236+58)]
}

# Calculate p-value for chiral proportion
prop_congloms_chiral = chiral_counts["Chiral"][0] / (1735+343)
prop_non_congloms_chiral = chiral_counts["Chiral"][1] / (18971+5122)

pooled_prop_chiral = (chiral_counts["Chiral"][0] + chiral_counts["Chiral"][1]) / ((1735+343) + (18971+5122))
std_error_chiral = np.sqrt(pooled_prop_chiral * (1 - pooled_prop_chiral) * (1/(1735+343) + 1/(18971+5122)))

z_stat_chiral = (prop_congloms_chiral - prop_non_congloms_chiral) / std_error_chiral
p_value_chiral = 1 - stats.norm.cdf(z_stat_chiral)  # One-tailed, higher in conglomerates

# Hypothesis 3: Proportion of Structures with Specific Compound Names (higher in non-conglomerates)

# Total counts after filtering Structures with Specific Compound Names
filtered_counts = {
    "Conglomerates": (1612+331),
    "Non-Conglomerates": (17346+4910)
}

# Total counts before filtering
total_counts_before = {
    "Conglomerates": (1735+343),
    "Non-Conglomerates": (18976+5123)
}

# Proportions with specific compound names
prop_congloms_filtered = filtered_counts["Conglomerates"] / total_counts_before["Conglomerates"]
prop_non_congloms_filtered = filtered_counts["Non-Conglomerates"] / total_counts_before["Non-Conglomerates"]

# Calculate p-value for the hypothesis that the proportion is higher in non-conglomerates
pooled_prop_filtered = (filtered_counts["Conglomerates"] + filtered_counts["Non-Conglomerates"]) / (total_counts_before["Conglomerates"] + total_counts_before["Non-Conglomerates"])
std_error_filtered = np.sqrt(pooled_prop_filtered * (1 - pooled_prop_filtered) * (1/total_counts_before["Conglomerates"] + 1/total_counts_before["Non-Conglomerates"]))

z_stat_filtered = (prop_non_congloms_filtered - prop_congloms_filtered) / std_error_filtered
p_value_filtered = 1 - stats.norm.cdf(z_stat_filtered)  # One-tailed, higher in non-conglomerates

# Print out all results
print("Keywords:", p_values_keywords, "Chirality:", p_value_chiral, "Compound Names:", p_value_filtered)
