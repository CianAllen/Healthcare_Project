# Healthcare Specialist Prediction Using BioClinical BERT

This project presents a complete end-to-end machine learning pipeline for predicting the most appropriate medical specialist from referral letters using state-of-the-art NLP techniques. It demonstrates the full data science workflow — from data exploration to model deployment — in a reproducible and production-ready format.

## Part 1: Healthcare Data Analysis
We begin with exploratory data analysis on a synthetic healthcare dataset of 55,500 patient records. This step involves statistical profiling, distribution analysis, and correlation studies to uncover patterns in demographics, medical conditions, and healthcare utilization. Key insights are stored in JSON format to guide the next phases.

## Part 2: Synthetic Data Generation
Using GPT-2, we generate 5,000 realistic synthetic referral letters that preserve clinical authenticity while avoiding privacy risks. These letters follow Canadian healthcare conventions and are automatically paired with the correct medical specialist, forming a robust labeled dataset for model training.

## Part 3: BioClinical BERT Classification
We fine-tune the BioClinical BERT model (`emilyalsentzer/Bio_ClinicalBERT`) to classify referral letters into specialist categories. The pipeline uses 5-fold cross-validation for reliable evaluation, achieving a mean accuracy of 63.52% and F1-score of 63.79%. The best-performing model and all configurations are saved for easy deployment.

This project is optimized for GPU execution in Google Colab and includes detailed metric tracking in JSON plus human-readable summaries. It highlights best practices in modern ML, including synthetic data creation, domain-specific transfer learning, rigorous validation, and reproducibility.

Whether you’re exploring healthcare NLP or building production-ready ML workflows, this repo provides a clear, practical example of applying deep learning to real-world medical text classification challenges.
