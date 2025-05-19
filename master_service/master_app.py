from flask import Flask, jsonify, request, Response
import subprocess
import re
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app) # otherwise stream is not returned at index or master (?) 
## code for https version if needed
# @app.before_request
# def redirect_to_https():
#     if not request.is_secure:
#         url = request.url.replace("http://", "https://", 1)
#         return jsonify({"message": "Redirecting to HTTPS", "redirect": url}), 301

#####
active_runs = {"run_id": None}
# Function to clean and escape the logs
def clean_log_output(log_text):
    """Remove ANSI color codes, emojis, and special characters."""
    # Remove ANSI escape sequences for colors
    clean_text = re.sub(r'\x1b\[[0-9;]*[mGKH]', '', log_text)
    # Remove emojis (any non-ASCII characters)
    clean_text = re.sub(r'[^\x00-\x7F]+', '', clean_text)  # Remove non-ASCII characters

    return clean_text

import re

def read_runid(log_text):
    match = re.search(r'run_id.*?(\d+)', log_text)
    if match:
        return match.group(1)
    return None

def format_flwr_input(data):
    for key, value in data.items():
        if isinstance(value,str):
            data[key]=f"'{value}'"
    return " ".join(f"{key}={value}" for key, value in data.items()) 


def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flatten a nested dictionary.

    Args:
        d (dict): The dictionary to flatten
        parent_key (str): The base key string (used in recursion)
        sep (str): Separator between keys (default is '.')

    Returns:
        dict: A flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



# Generator to stream the Flower logs
def generate_flwr_logs(inputs):
    """Stream Flower logs in real-time, cleaning them before sending."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1" 
    #inputs='epochs=1'
    process = subprocess.Popen(
        ['flwr', 'run', 'flower-app', 'local-deployment-docker','--run-config',f'{inputs}','--stream'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,
        env=env  # Line-buffered output
    )
    for line in iter(process.stdout.readline, ''):
        run_id_match = re.search(r'run.*?(\d+)', line) # catch run_id from stream
        if run_id_match:
            active_runs["run_id"] = run_id_match.group(1)
            print(f"[INFO] Captured run_id: {active_runs['run_id']}")
        yield line  # Send each cleaned line as it appears
    process.stdout.close()
    process.wait()

# Route to start the Flower run and stream logs

@app.route('/flwr-run', methods=['POST'])
def flwr_run():
    """Start Flower run and stream cleaned logs."""
    try:
        data = request.get_json()  # Extract JSON data from request
        if not data:
            return "Invalid input data", 400
        
        flattened = flatten_dict(data)
        
        inputs = format_flwr_input(flattened)  # Convert JSON to CLI args
        return Response(generate_flwr_logs(inputs), mimetype='text/plain')
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/flwr-stop', methods=['POST'])
def flwr_stop():
    try:
        data = request.get_json()  # Get the request data
        run_id = data.get("run_id")
        if not run_id:
            return jsonify({"message": f"No active run with id:{run_id} to stop"}), 400
        
        env = os.environ.copy()
        process = subprocess.Popen(
            ['flwr', 'stop', f'{run_id}','flower-app','local-deployment-docker'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        output, _ = process.communicate(timeout=10)
        return jsonify({"message": f"Run {run_id} stopped successfully", "log": output}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# def ls_flwr():
#     return

# @app.route('/flwr-log', methods=['POST'])
# def log_flwr():
#     return

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)# , ssl_context=('cert.pem', 'key.pem'))
## mount certificates to master docker