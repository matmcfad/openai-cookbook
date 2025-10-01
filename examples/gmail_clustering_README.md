# Gmail Email Clustering Script

This script fetches emails from your Gmail inbox, generates embeddings using OpenAI's API, performs k-means clustering to group similar emails together, and outputs the results with AI-generated cluster themes.

## Features

- Fetches the last 500 emails (configurable) from your Gmail inbox
- Generates embeddings using OpenAI's embedding API
- Performs k-means clustering to identify email groups
- Uses GPT to generate descriptive themes for each cluster
- Creates a t-SNE visualization of the clusters
- Outputs results to CSV with cluster assignments and themes
- Displays sample emails from each cluster

## Prerequisites

### 1. Python Packages

Install the required packages:

```bash
pip install openai pandas numpy scikit-learn matplotlib google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### 2. OpenAI API Key

You need an OpenAI API key. Set it as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or on Windows:
```cmd
set OPENAI_API_KEY=your-api-key-here
```

### 3. Gmail API Credentials

You need to set up Gmail API access:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select an existing one)
3. Enable the Gmail API:
   - Go to "APIs & Services" > "Library"
   - Search for "Gmail API"
   - Click "Enable"
4. Create OAuth 2.0 credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Choose "Desktop app" as the application type
   - Download the credentials file
5. Save the downloaded file as `credentials.json` in the same directory as the script

## Usage

### Basic Usage

```bash
python gmail_clustering.py
```

This will:
- Fetch the last 500 emails from your inbox
- Create 5 clusters
- Save results to `email_clusters.csv`
- Create a visualization in `email_clusters.png`

### Advanced Options

```bash
python gmail_clustering.py --max-emails 300 --n-clusters 4 --output my_results.csv
```

#### Command-line Arguments

- `--max-emails N`: Number of emails to fetch (default: 500)
- `--n-clusters K`: Number of clusters to create (default: 5)
- `--output FILE`: Output CSV filename (default: email_clusters.csv)
- `--no-viz`: Skip visualization generation

## First Run

On the first run, the script will:
1. Open your browser for Gmail authentication
2. Ask you to authorize the application
3. Save a `token.pickle` file for future runs

After the first run, you won't need to authenticate again unless the token expires.

## Output

### CSV File

The output CSV contains:
- `id`: Gmail message ID
- `subject`: Email subject
- `from`: Sender email address
- `date`: Email date
- `body`: Preview of email body (first 500 characters)
- `Cluster`: Cluster number (0 to K-1)
- `ClusterTheme`: AI-generated description of the cluster theme

### Visualization

A PNG file showing clusters in 2D space using t-SNE dimensionality reduction. Different colors represent different clusters.

### Console Output

The script prints:
- Progress updates during email fetching and processing
- Cluster themes for each group
- Sample emails from each cluster

## Example Output

```
Cluster 0 (87 emails): Social media notifications and updates from various platforms like Facebook, Twitter, and LinkedIn.

Sample 1:
  Subject: You have 5 new notifications on Facebook
  From: notification@facebook.com
  Date: Mon, 15 Jan 2024 10:30:00 -0800
  Preview: Your friend John posted a new photo...

Cluster 1 (123 emails): Work-related project updates, meeting invites, and team communications.
...
```

## Troubleshooting

### "Error: credentials.json not found"
Make sure you've downloaded the OAuth credentials from Google Cloud Console and saved them as `credentials.json`.

### "Error: OPENAI_API_KEY environment variable not set"
Set your OpenAI API key as an environment variable before running the script.

### Rate Limits
If you hit OpenAI API rate limits, the script may fail. Try:
- Reducing `--max-emails`
- Waiting a few minutes and trying again
- Upgrading your OpenAI API plan

### Gmail API Quota
Gmail API has daily quotas. If you exceed them, wait 24 hours or request a quota increase.

## Cost Estimation

### OpenAI API Costs
- Embeddings: ~$0.00002 per email (using text-embedding-3-small)
- GPT calls: ~$0.0005 per cluster (using gpt-3.5-turbo)

For 500 emails with 5 clusters:
- Embeddings: ~$0.01
- GPT: ~$0.0025
- **Total: ~$0.01-0.02**

## Privacy and Security

- Your Gmail credentials are stored locally in `token.pickle`
- Emails are processed locally and only sent to OpenAI for embedding/clustering
- No emails are stored on external servers (except OpenAI's temporary processing)
- The script only requests read-only access to your Gmail

## Related Examples

This script is based on the clustering examples in the OpenAI Cookbook:
- [Clustering.ipynb](Clustering.ipynb) - Basic k-means clustering example
- [Clustering_for_transaction_classification.ipynb](Clustering_for_transaction_classification.ipynb) - Transaction clustering

## License

This script is part of the OpenAI Cookbook and follows the same license.
