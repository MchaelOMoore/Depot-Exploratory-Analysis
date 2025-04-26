Depot Incident Analysis
Data-driven analysis of railway depot safety incidents to identify trends, measure harm, and support industry-wide safety improvements.

Overview
This project analyses workforce safety incidents in railway depots across Great Britain, using anonymised data from the Safety Management Intelligence System (SMIS). It was developed to support the Depot Working Group (DWG) and Rail Safety and Standards Board (RSSB) initiatives aimed at reducing depot-related harm.

Project Goals
 - Investigate trends in depot safety incidents.
 - Measure the impact of new safety standards.
 - Identify key risk factors contributing to workforce harm.

Dataset
  - Source: Adapted from SMIS (anonymised for confidentiality).
  - Key Fields: Incident ID, Date, Period, Location, Incident Type, Injury Severity.
  - Metric: Fatalities and Weighted Injuries (FWI) Index — a severity-weighted score for workforce harm.

Methodology
  - Data Cleaning: Merged fields, removed duplicates, standardised incident categories.
  - Trend Analysis:
    - Stacked column charts with 13-period moving averages.
    - Confidence intervals to assess statistical significance.

  - Severity Analysis:
    - Calculated FWI per incident.
    - Grouped injuries into simplified categories.
    - Visualised with line and stacked column graphs.

Project Structure

C:.
|   depot_incident_analysis.pdf    # Project report
|   
+---csv                            # Data files (anonymised SMIS extracts)
|       all_incidents.csv
|       fwi_per_incident.csv
|       injury_incidents.csv
|       
\---Python Scripts                 # Jupyter notebooks for analysis
        All_incidents_notebook.ipynb
        fwi_per_incident_notebook.ipynb
        injury_incidents_notebook.ipynb

Key Results
  - 📉 Incident Reduction: Statistically significant decrease in depot incidents from mid-2023 onward.
  - 🔵 Injury Insights: 'Specified' injuries contributed most to overall harm.
  - 📈 FWI Trends: Decrease in average harm per incident after late 2023.

Challenges
  - Data quality issues (missing values, inconsistent reporting).
  - Categorisation complexity.
  - Variance in incident reporting practices.

Future Improvements
  - Machine learning for smarter incident categorisation.
  - Predictive modelling to forecast high-risk periods.
  - Granular risk analysis by role and depot location.
  - Real-time automated safety dashboards.

Conclusion
This project highlights the power of data-driven decision making in improving depot worker safety. Future work will focus  on predictive analytics and better automation to further enhance workplace safety in railway depots.

















