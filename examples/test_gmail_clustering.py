#!/usr/bin/env python3
"""
Test script to demonstrate the Gmail clustering functionality with sample data.
This allows testing the clustering logic without requiring Gmail API credentials.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the utils directory to the path
sys.path.append(str(Path(__file__).parent / "utils"))

# Set a dummy API key for testing
os.environ['OPENAI_API_KEY'] = 'dummy-key-for-testing'

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Sample email data (simulating Gmail emails)
sample_emails = [
    {
        'id': '1',
        'subject': 'Team Meeting Tomorrow',
        'from': 'manager@company.com',
        'date': 'Mon, 15 Jan 2024 09:00:00',
        'body': 'Hi team, reminder about our weekly sync meeting tomorrow at 10am.',
        'combined': 'Subject: Team Meeting Tomorrow\n\nHi team, reminder about our weekly sync meeting tomorrow at 10am.'
    },
    {
        'id': '2',
        'subject': 'Your LinkedIn notification',
        'from': 'notifications@linkedin.com',
        'date': 'Mon, 15 Jan 2024 10:30:00',
        'body': 'John viewed your profile. You have 3 new connection requests.',
        'combined': 'Subject: Your LinkedIn notification\n\nJohn viewed your profile. You have 3 new connection requests.'
    },
    {
        'id': '3',
        'subject': 'Project Update',
        'from': 'colleague@company.com',
        'date': 'Mon, 15 Jan 2024 11:00:00',
        'body': 'The new feature is ready for review. Please check the PR.',
        'combined': 'Subject: Project Update\n\nThe new feature is ready for review. Please check the PR.'
    },
    {
        'id': '4',
        'subject': 'Facebook notification',
        'from': 'notify@facebook.com',
        'date': 'Mon, 15 Jan 2024 12:00:00',
        'body': 'Sarah posted a new photo. You have 5 friend requests.',
        'combined': 'Subject: Facebook notification\n\nSarah posted a new photo. You have 5 friend requests.'
    },
    {
        'id': '5',
        'subject': 'Sprint Planning',
        'from': 'scrum@company.com',
        'date': 'Mon, 15 Jan 2024 13:00:00',
        'body': 'Sprint planning session scheduled for Wednesday 2pm.',
        'combined': 'Subject: Sprint Planning\n\nSprint planning session scheduled for Wednesday 2pm.'
    },
]

# Duplicate emails to have more data
for i in range(len(sample_emails)):
    for j in range(20):
        new_email = sample_emails[i].copy()
        new_email['id'] = f"{new_email['id']}-{j}"
        sample_emails.append(new_email)

print(f"Testing with {len(sample_emails)} sample emails")
print("="*80)

df = pd.DataFrame(sample_emails[:len(sample_emails)])

# Generate dummy embeddings (in real script, these come from OpenAI)
print("Generating dummy embeddings...")
np.random.seed(42)
# Create embeddings that will cluster work emails together and social media together
embeddings = []
for email in sample_emails[:len(sample_emails)]:
    if 'notification' in email['subject'].lower() or 'facebook' in email['from'].lower() or 'linkedin' in email['from'].lower():
        # Social media cluster
        base = np.array([0.8, 0.2] + [0.0]*1534)
    else:
        # Work emails cluster
        base = np.array([0.2, 0.8] + [0.0]*1534)
    
    # Add some noise
    noise = np.random.normal(0, 0.05, 1536)
    embeddings.append(base + noise)

# Perform clustering
print("Performing k-means clustering...")
matrix = np.array(embeddings)
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
kmeans.fit(matrix)
labels = kmeans.labels_
df['Cluster'] = labels

# Show results
print("\n" + "="*80)
print("CLUSTERING RESULTS")
print("="*80)

for i in range(n_clusters):
    cluster_emails = df[df.Cluster == i]
    print(f"\nCluster {i}: {len(cluster_emails)} emails")
    print("-"*80)
    
    # Show first 3 unique emails in this cluster
    seen_subjects = set()
    count = 0
    for _, row in cluster_emails.iterrows():
        if row['subject'] not in seen_subjects:
            seen_subjects.add(row['subject'])
            print(f"  Subject: {row['subject']}")
            print(f"  From: {row['from']}")
            print(f"  Body: {row['body'][:80]}...")
            print()
            count += 1
            if count >= 3:
                break

# Create visualization
print("Creating visualization...")
# Use PCA for visualization since we have low dimensional embeddings
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
vis_dims = pca.fit_transform(matrix)

x = [x for x, y in vis_dims]
y = [y for x, y in vis_dims]

plt.figure(figsize=(10, 6))

colors = ["blue", "orange"]
for category in range(n_clusters):
    color = colors[category % len(colors)]
    xs = np.array(x)[df.Cluster == category]
    ys = np.array(y)[df.Cluster == category]
    plt.scatter(xs, ys, color=color, alpha=0.5, label=f'Cluster {category}', s=50)
    
    avg_x = xs.mean()
    avg_y = ys.mean()
    plt.scatter(avg_x, avg_y, marker="x", color=color, s=200, linewidths=3)

plt.title("Email Clusters Visualization (Test Data)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('test_email_clusters.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved to test_email_clusters.png")

# Save results
output_file = 'test_email_clusters.csv'
df[['id', 'subject', 'from', 'date', 'body', 'Cluster']].to_csv(output_file, index=False)
print(f"✓ Results saved to {output_file}")

print("\n" + "="*80)
print("TEST COMPLETE!")
print("="*80)
print("\nThis demonstrates the clustering logic that the main script uses.")
print("The actual script fetches real emails from Gmail and uses OpenAI for embeddings.")
