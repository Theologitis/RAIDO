# # from flask import Flask, jsonify, request
# # import subprocess
# # from flask_cors import CORS  # Import CORS

# # app = Flask(__name__)
# # CORS(app)

# # @app.route('/run-flwr', methods=['POST'])
# # def run_flwr():
# #     try:
# #         # Run the flwr command as a subprocess
# #         result = subprocess.run(['flwr', 'run', 'fl-tabular', 'local-deployment-docker', '--stream'],
# #                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# #         # Capture stdout and stderr
# #         output = result.stdout + '\n' + result.stderr
# #         return jsonify({'status': 'success', 'output': output}), 200
# #     except Exception as e:
# #         return jsonify({'status': 'error', 'message': str(e)}), 500

# # if __name__ == '__main__':
# #     app.run(host='0.0.0.0', port=5000)

# ## FOR REAL TIME PROGRESS MESSAGES ##
# from flask import Flask, jsonify, request, Response
# import subprocess
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# def generate_flwr_logs():
#     """Stream Flower logs in real-time."""
#     process = subprocess.Popen(
#         ['flwr', 'run', 'flower-app', 'local-deployment-docker', '--stream'],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT,  # Merge stderr into stdout
#         text=True,
#         bufsize=1,  # Line-buffered output
#     )
#     for line in iter(process.stdout.readline, ''):
#         yield line  # Send each line as it appears
#     process.stdout.close()
#     process.wait()

# @app.route('/run-flwr', methods=['POST'])
# def run_flwr():
#     """Start Flower run and stream logs."""
#     return Response(generate_flwr_logs(), mimetype='text/plain')

# # @app.route('/flwr-ls',methods=['POST'])
# # def ls_flwr():
# #     return

# # @app.route('/flwr-log',methods=['POST'])
# # def log_flwr():
# #     return

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, threaded=True)
from flask import Flask, jsonify, request, Response
import subprocess
import re
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app) # otherwise stream is not returned at index or master (?) 
# @app.before_request
# def redirect_to_https():
#     if not request.is_secure:
#         url = request.url.replace("http://", "https://", 1)
#         return jsonify({"message": "Redirecting to HTTPS", "redirect": url}), 301
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

def flwr_stop(run_id,env):
    process = subprocess.Popen(
        ['flwr', 'stop', f'{run_id}'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,
        env=env  # Line-buffered output
    )

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
        clean_line = clean_log_output(line)  # Clean the log line before sending
        yield line  # Send each cleaned line as it appears
    process.stdout.close()
    process.wait()

    # try:
    #     for line in iter(process.stdout.readline, ''):
    #         if request.environ.get('werkzeug.server.shutdown'):  # Detect if Flask is shutting down
    #             break
    #         clean_line = clean_log_output(line)
    #         yield clean_line
    # finally:
    
    #     process.stdout.close()
    #     process.terminate()  # Gracefully stop the process
    #     process.wait()

# Route to start the Flower run and stream logs
@app.route('/run-flwr', methods=['POST'])
def run_flwr():
    """Start Flower run and stream cleaned logs."""
    try:
        data = request.get_json()  # Extract JSON data from request
        if not data:
            return "Invalid input data", 400

        inputs = format_flwr_input(data)  # Convert JSON to CLI args
        return Response(generate_flwr_logs(inputs), mimetype='text/plain')
    except Exception as e:
        return f"Error: {str(e)}", 500

# def run_flwr():
#     # data = request.get_json()  # Extract JSON data from reques
#     data = {"epochs": 5, "learning_rate": 0.01, "batch_size": 32}
#     input=format_flwr_input(data)
#     return Response(generate_flwr_logs(input), mimetype='text/plain')
# @app.route('/flwr-ls', methods=['POST'])
# def ls_flwr():
#     return

# @app.route('/flwr-log', methods=['POST'])
# def log_flwr():
#     return

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)# , ssl_context=('cert.pem', 'key.pem'))
## mount certificates to master docker