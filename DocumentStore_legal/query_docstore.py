import subprocess
import urllib.parse

def execute_curl_request(query, k=2):
    # URL encode the query
    encoded_query = urllib.parse.quote(query)
    url = f"http://localhost:8001/v1/retrieve?query={encoded_query}&k={k}"
    
    # Construct the curl command
    command = [
        "curl", "-X", "GET", url, "-H", "accept: */*"
    ]
    
    # Execute the command
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Response:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)

# Example usage
query = "How many shares were repurchased since the inception of the share repurchase program, and what was the total cost?"
top_k=5

execute_curl_request(query=query, k=top_k)
