import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import re

# 1. Log Generation (for demonstration purposes)
def generate_sample_logs(num_normal=200, num_anomalies=10):
    """Generate synthetic log data with normal and anomalous patterns."""
    np.random.seed(42)
    
    # Create a base timestamp
    base_time = datetime(2025, 3, 10, 8, 0, 0)
    
    # Generate normal logs - regular login patterns
    logs = []
    users = ["alice", "bob", "charlie", "dave", "admin"]
    
    for i in range(num_normal):
        # Random time increment (1-60 seconds)
        time_increment = np.random.randint(1, 60)
        timestamp = base_time + timedelta(seconds=i*time_increment)
        
        # Randomly select user and log level
        user = np.random.choice(users)
        level = np.random.choice(["INFO", "INFO", "INFO", "WARN", "ERROR"], p=[0.7, 0.1, 0.1, 0.05, 0.05])
        
        # Generate message based on level
        if level == "INFO":
            message = np.random.choice([
                "User login success",
                "Page accessed",
                "File downloaded",
                "Report generated"
            ])
        elif level == "WARN":
            message = np.random.choice([
                "Slow response time",
                "Resource usage high",
                "Session timeout"
            ])
        else:  # ERROR
            message = np.random.choice([
                "Failed login attempt",
                "Database connection error",
                "Permission denied"
            ])
        
        # Create log entry
        log_entry = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}, {level}, {message}, user: {user}"
        logs.append(log_entry)
    
    # Generate anomalous logs - burst of failed login attempts (simulating brute force)
    anomaly_time = base_time + timedelta(minutes=45)  # Place anomalies later in the timeline
    attacker_user = np.random.choice(users)
    
    for i in range(num_anomalies):
        timestamp = anomaly_time + timedelta(seconds=i*2)  # Rapid succession (every 2 seconds)
        level = "ERROR"
        message = "Failed login attempt"
        log_entry = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}, {level}, {message}, user: {attacker_user}"
        logs.append(log_entry)
    
    return logs

# 2. Log Parsing
def parse_logs(logs):
    """Parse raw log entries into structured data."""
    parsed_logs = []
    
    for line in logs:
        parts = [p.strip() for p in line.split(",")]
        timestamp = parts[0]
        level = parts[1]
        message = parts[2]
        user = parts[3].split(":")[1].strip() if len(parts) > 3 and "user:" in parts[3] else None
        
        parsed_logs.append({
            "timestamp": timestamp, 
            "level": level, 
            "message": message, 
            "user": user
        })
    
    # Convert to DataFrame
    return pd.DataFrame(parsed_logs)

# 3. Feature Extraction
def extract_features(df):
    """Extract numeric features from log data for anomaly detection."""
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by minute and user, count events (especially errors)
    df['minute'] = df['timestamp'].dt.floor('min')
    
    # Count login failures per minute per user
    login_failures = df[df['message'] == 'Failed login attempt'].groupby(['minute', 'user']).size().reset_index(name='failures')
    
    # Add total events per minute per user
    total_events = df.groupby(['minute', 'user']).size().reset_index(name='total_events')
    
    # Merge the features
    features = pd.merge(total_events, login_failures, on=['minute', 'user'], how='left')
    features['failures'] = features['failures'].fillna(0)
    
    # Calculate ratio of failures to total events (where total > 0)
    features['failure_ratio'] = features.apply(
        lambda row: row['failures'] / row['total_events'] if row['total_events'] > 0 else 0, 
        axis=1
    )
    
    return features

# 4. Anomaly Detection
def detect_anomalies(features, contamination=0.05):
    """Use Isolation Forest to detect anomalies in the feature data."""
    # Select numeric features for detection
    X = features[['failures', 'total_events', 'failure_ratio']].values
    
    # Initialize and train the model
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)
    
    # Predict anomalies
    features['anomaly'] = model.predict(X)
    features['anomaly_score'] = model.decision_function(X)
    
    # Flag anomalies (anomaly == -1 means it's an anomaly)
    anomalies = features[features['anomaly'] == -1].sort_values('anomaly_score')
    
    return anomalies, features

# 5. Visualization
def visualize_results(features, anomalies):
    """Visualize the anomaly detection results."""
    plt.figure(figsize=(12, 6))
    
    # Plot normal events
    normal = features[features['anomaly'] == 1]
    plt.scatter(normal['minute'], normal['failures'], c='blue', label='Normal Events', alpha=0.5)
    
    # Plot anomalies
    plt.scatter(anomalies['minute'], anomalies['failures'], c='red', label='Anomalies', s=100, marker='X')
    
    plt.title('Failed Login Attempts Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Failed Attempts')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 6. Automated Response Simulation
def simulate_response(anomalies):
    """Simulate automated response to detected anomalies."""
    if len(anomalies) > 0:
        print("\n===== AUTOMATED RESPONSE TRIGGERED =====")
        for _, anomaly in anomalies.iterrows():
            user = anomaly['user']
            minute = anomaly['minute']
            failures = anomaly['failures']
            score = anomaly['anomaly_score']
            
            risk_level = "HIGH" if score < -0.3 else "MEDIUM"
            
            print(f"ALERT! {risk_level} RISK - User '{user}' had {int(failures)} failed login attempts at {minute}")
            print(f"Response: {'Account temporarily locked' if risk_level == 'HIGH' else 'Additional authentication required'}")
            print(f"Security team notified. Incident ID: SEC-{hash(str(minute)+user) % 1000000:06d}")
            print("-----------------------------------------")

# Main execution
def main():
    print("Starting AI-powered SIEM Log Analysis System...\n")
    
    # Step 1: Generate or load logs
    logs = generate_sample_logs(num_normal=200, num_anomalies=15)
    print(f"Processing {len(logs)} log entries...\n")
    
    # Display a few sample logs
    print("Sample logs:")
    for log in logs[:5]:
        print(f"  {log}")
    print("  ...\n")
    
    # Step 2: Parse logs
    df_logs = parse_logs(logs)
    print(f"Parsed {len(df_logs)} log entries into structured format.")
    
    # Step 3: Extract features
    features = extract_features(df_logs)
    print(f"Extracted features for {len(features)} time-user combinations.\n")
    
    # Step 4: Detect anomalies
    anomalies, features_with_predictions = detect_anomalies(features)
    print(f"Detected {len(anomalies)} potential anomalies.\n")
    
    # Step 5: Report findings
    print("Analysis Summary:")
    total_users = len(features['user'].unique())
    total_minutes = len(features['minute'].unique())
    print(f"  Monitored {total_users} users over approximately {total_minutes} minutes")
    print(f"  Found {len(anomalies)} suspicious activity patterns\n")
    
    # Step 6: Simulate response
    simulate_response(anomalies)
    
    # Step 7: Visualization would go here in an interactive environment
    # visualize_results(features_with_predictions, anomalies)
    
    print("\nSIEM Analysis complete!")

if __name__ == "__main__":
    main()
