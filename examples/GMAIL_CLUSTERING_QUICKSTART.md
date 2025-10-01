# Gmail Email Clustering - Quick Start Guide

This guide shows you how to use the Gmail email clustering script.

## What This Script Does

The script takes your last 500 emails from Gmail, uses OpenAI to understand their content, groups similar emails together using k-means clustering, and provides insights about each group.

## Quick Test (No Gmail Required)

To see how the clustering works without setting up Gmail API:

```bash
cd examples
python test_gmail_clustering.py
```

This will:
- Create sample emails (work emails vs social media notifications)
- Cluster them into groups
- Save results to `test_email_clusters.csv`
- Create a visualization in `test_email_clusters.png`

## Full Setup for Real Gmail Emails

### 1. Install Dependencies

```bash
pip install openai pandas numpy scikit-learn matplotlib \
            google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY='your-openai-api-key'
```

### 3. Set Up Gmail API

1. Go to https://console.cloud.google.com/
2. Create a new project
3. Enable Gmail API
4. Create OAuth 2.0 credentials (Desktop app)
5. Download and save as `credentials.json` in the examples directory

### 4. Run the Script

```bash
cd examples
python gmail_clustering.py
```

On first run, it will open your browser to authorize Gmail access.

## Example Commands

### Cluster last 300 emails into 4 groups:
```bash
python gmail_clustering.py --max-emails 300 --n-clusters 4
```

### Save to specific file:
```bash
python gmail_clustering.py --output my_clusters.csv
```

### Skip visualization:
```bash
python gmail_clustering.py --no-viz
```

## What You'll Get

1. **CSV file** (`email_clusters.csv`) with:
   - Email subjects, senders, dates
   - Cluster assignments
   - AI-generated cluster themes

2. **Visualization** (`email_clusters.png`):
   - 2D plot of email clusters
   - Different colors for each cluster

3. **Console output**:
   - Cluster themes
   - Sample emails from each cluster
   - Summary statistics

## Example Output

```
Cluster 0 (124 emails): Social media notifications from Facebook, LinkedIn, and Twitter

Sample 1:
  Subject: You have 3 new notifications
  From: notify@facebook.com
  ...

Cluster 1 (87 emails): Work-related project updates and team communications

Sample 1:
  Subject: Sprint Review Meeting
  From: manager@company.com
  ...
```

## Cost Estimate

For 500 emails:
- Embeddings: ~$0.01
- GPT cluster descriptions: ~$0.0025
- **Total: ~$0.01-0.02**

## Troubleshooting

See the full [README](gmail_clustering_README.md) for detailed troubleshooting.

### Common Issues:

**"credentials.json not found"**
- Download OAuth credentials from Google Cloud Console

**"OPENAI_API_KEY not set"**
- Set the environment variable: `export OPENAI_API_KEY='your-key'`

**Rate limits**
- Reduce `--max-emails` or wait a few minutes

## Learn More

This script is based on the clustering examples in the OpenAI Cookbook:
- [Clustering.ipynb](Clustering.ipynb)
- [Clustering_for_transaction_classification.ipynb](Clustering_for_transaction_classification.ipynb)
