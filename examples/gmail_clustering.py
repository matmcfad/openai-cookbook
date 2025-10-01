#!/usr/bin/env python3
"""
Gmail Email Clustering Script

This script fetches the last 500 emails from a Gmail inbox, generates embeddings using OpenAI,
performs k-means clustering, and outputs cluster themes and sample emails.

Requirements:
- OpenAI API key (set as OPENAI_API_KEY environment variable)
- Gmail API credentials (credentials.json file)
- Python packages: google-auth, google-auth-oauthlib, google-auth-httplib2, google-api-python-client,
                   openai, pandas, numpy, scikit-learn, matplotlib

Usage:
    python gmail_clustering.py [--max-emails N] [--n-clusters K] [--output FILE]
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any
import base64
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from openai import OpenAI

# Add the utils directory to the path
sys.path.append(str(Path(__file__).parent / "utils"))
from embeddings_utils import get_embeddings

# Gmail API imports
GMAIL_AVAILABLE = True
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    GMAIL_AVAILABLE = False


# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def authenticate_gmail():
    """Authenticate and return Gmail API service."""
    creds = None
    token_path = 'token.pickle'
    creds_path = 'credentials.json'
    
    # Check if token.pickle exists
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(creds_path):
                print(f"Error: {creds_path} not found.")
                print("Please download your OAuth 2.0 credentials from Google Cloud Console:")
                print("1. Go to https://console.cloud.google.com/")
                print("2. Create a project and enable Gmail API")
                print("3. Create OAuth 2.0 credentials (Desktop app)")
                print("4. Download the credentials and save as 'credentials.json'")
                sys.exit(1)
            
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
    
    return build('gmail', 'v1', credentials=creds)


def get_email_text(payload):
    """Extract text content from email payload."""
    text = ""
    
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                if 'data' in part['body']:
                    text += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
            elif part['mimeType'] == 'text/html' and not text:
                # Only use HTML if no plain text is available
                if 'data' in part['body']:
                    html_text = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                    # Simple HTML tag removal
                    text += re.sub(r'<[^>]+>', '', html_text)
            elif 'parts' in part:
                # Recursive call for nested parts
                text += get_email_text(part)
    else:
        if 'data' in payload['body']:
            text = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
    
    return text


def fetch_emails(service, max_results=500):
    """Fetch emails from Gmail inbox."""
    print(f"Fetching last {max_results} emails from inbox...")
    
    emails = []
    
    try:
        # Get message IDs
        results = service.users().messages().list(
            userId='me',
            labelIds=['INBOX'],
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])
        
        if not messages:
            print("No messages found.")
            return pd.DataFrame()
        
        print(f"Found {len(messages)} emails. Fetching details...")
        
        # Fetch full message details
        for i, message in enumerate(messages):
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(messages)} emails...")
            
            msg = service.users().messages().get(
                userId='me',
                id=message['id'],
                format='full'
            ).execute()
            
            headers = msg['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            from_email = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')
            
            # Extract email body
            body = get_email_text(msg['payload'])
            
            # Combine subject and body for embedding
            combined_text = f"Subject: {subject}\n\n{body[:1000]}"  # Limit body to 1000 chars
            
            emails.append({
                'id': message['id'],
                'subject': subject,
                'from': from_email,
                'date': date,
                'body': body[:500],  # Store first 500 chars for display
                'combined': combined_text
            })
        
        print(f"Successfully fetched {len(emails)} emails.")
        
    except HttpError as error:
        print(f'An error occurred: {error}')
        return pd.DataFrame()
    
    return pd.DataFrame(emails)


def generate_embeddings(df, model="text-embedding-3-small"):
    """Generate embeddings for email texts."""
    print("Generating embeddings using OpenAI API...")
    
    texts = df['combined'].tolist()
    
    # Process in batches to avoid rate limits
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
        embeddings = get_embeddings(batch, model=model)
        all_embeddings.extend(embeddings)
    
    print("Embeddings generated successfully.")
    return all_embeddings


def perform_clustering(embeddings, n_clusters=4):
    """Perform k-means clustering on embeddings."""
    print(f"Performing k-means clustering with {n_clusters} clusters...")
    
    matrix = np.array(embeddings)
    
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(matrix)
    labels = kmeans.labels_
    
    print("Clustering completed.")
    return labels, matrix


def visualize_clusters(df, matrix, output_file='email_clusters.png'):
    """Create t-SNE visualization of clusters."""
    print("Creating cluster visualization...")
    
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    vis_dims = tsne.fit_transform(matrix)
    
    x = [x for x, y in vis_dims]
    y = [y for x, y in vis_dims]
    
    plt.figure(figsize=(12, 8))
    
    colors = ["purple", "green", "red", "blue", "orange", "brown", "pink", "gray", "olive", "cyan"]
    n_clusters = df['Cluster'].nunique()
    
    for category in range(n_clusters):
        color = colors[category % len(colors)]
        xs = np.array(x)[df.Cluster == category]
        ys = np.array(y)[df.Cluster == category]
        plt.scatter(xs, ys, color=color, alpha=0.3, label=f'Cluster {category}')
        
        avg_x = xs.mean()
        avg_y = ys.mean()
        plt.scatter(avg_x, avg_y, marker="x", color=color, s=200)
    
    plt.title("Email Clusters Visualized in 2D using t-SNE")
    plt.legend()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    plt.close()


def generate_cluster_themes(df, n_clusters, client, samples_per_cluster=5):
    """Generate themes for each cluster using GPT."""
    print("\nGenerating cluster themes using GPT...")
    
    themes = []
    
    for i in range(n_clusters):
        cluster_emails = df[df.Cluster == i]
        
        # Sample emails from cluster
        sample_emails = cluster_emails.sample(
            min(samples_per_cluster, len(cluster_emails)),
            random_state=42
        )
        
        # Create prompt
        email_summaries = "\n".join([
            f"{j+1}. Subject: {row['subject']}\n   From: {row['from']}\n   Snippet: {row['body'][:100]}..."
            for j, (_, row) in enumerate(sample_emails.iterrows())
        ])
        
        messages = [
            {
                "role": "user",
                "content": f"What is the common theme among these emails? Provide a brief, descriptive theme (1-2 sentences).\n\nEmails:\n{email_summaries}\n\nTheme:"
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            max_tokens=100
        )
        
        theme = response.choices[0].message.content.strip()
        themes.append(theme)
        
        print(f"\nCluster {i} ({len(cluster_emails)} emails): {theme}")
    
    return themes


def save_results(df, themes, output_file='email_clusters.csv'):
    """Save clustering results to CSV."""
    print(f"\nSaving results to {output_file}...")
    
    # Add theme to dataframe
    df['ClusterTheme'] = df['Cluster'].apply(lambda x: themes[x] if x < len(themes) else "Unknown")
    
    # Save to CSV (without embeddings to keep file size manageable)
    df[['id', 'subject', 'from', 'date', 'body', 'Cluster', 'ClusterTheme']].to_csv(
        output_file,
        index=False
    )
    
    print(f"Results saved to {output_file}")


def print_cluster_summary(df, themes, samples_per_cluster=3):
    """Print a summary of each cluster."""
    print("\n" + "="*100)
    print("CLUSTER SUMMARY")
    print("="*100)
    
    for i in range(len(themes)):
        cluster_emails = df[df.Cluster == i]
        print(f"\n{'='*100}")
        print(f"Cluster {i}: {themes[i]}")
        print(f"Number of emails: {len(cluster_emails)}")
        print(f"{'='*100}")
        
        # Show sample emails
        samples = cluster_emails.sample(min(samples_per_cluster, len(cluster_emails)), random_state=42)
        
        for j, (_, row) in enumerate(samples.iterrows()):
            print(f"\nSample {j+1}:")
            print(f"  Subject: {row['subject']}")
            print(f"  From: {row['from']}")
            print(f"  Date: {row['date']}")
            print(f"  Preview: {row['body'][:150]}...")
        
        print("\n" + "-"*100)


def main():
    parser = argparse.ArgumentParser(
        description='Cluster Gmail emails using k-means and OpenAI embeddings'
    )
    parser.add_argument(
        '--max-emails',
        type=int,
        default=500,
        help='Maximum number of emails to fetch (default: 500)'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=5,
        help='Number of clusters for k-means (default: 5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='email_clusters.csv',
        help='Output CSV file name (default: email_clusters.csv)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    
    args = parser.parse_args()
    
    # Check for Gmail API libraries
    if not GMAIL_AVAILABLE:
        print("Error: Gmail API libraries not installed. Please install them with:")
        print("pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
        sys.exit(1)
    
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Authenticate Gmail
    service = authenticate_gmail()
    
    # Fetch emails
    df = fetch_emails(service, max_results=args.max_emails)
    
    if df.empty:
        print("No emails to process. Exiting.")
        sys.exit(1)
    
    # Generate embeddings
    embeddings = generate_embeddings(df)
    
    # Perform clustering
    labels, matrix = perform_clustering(embeddings, n_clusters=args.n_clusters)
    df['Cluster'] = labels
    
    # Generate cluster themes
    themes = generate_cluster_themes(df, args.n_clusters, client)
    
    # Save results
    save_results(df, themes, output_file=args.output)
    
    # Create visualization
    if not args.no_viz:
        viz_file = args.output.replace('.csv', '.png')
        visualize_clusters(df, matrix, output_file=viz_file)
    
    # Print summary
    print_cluster_summary(df, themes)
    
    print("\n" + "="*100)
    print("CLUSTERING COMPLETE!")
    print("="*100)
    print(f"Results saved to: {args.output}")
    if not args.no_viz:
        print(f"Visualization saved to: {viz_file}")


if __name__ == "__main__":
    main()
