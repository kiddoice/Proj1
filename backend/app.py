from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/detect_chicken', methods=['POST'])
def detect_chicken():
    # Here we would implement the logic for chicken detection
    data = request.get_json()
    # Process the data and perform detection
    result = {'message': 'Chicken detected!'}  # Placeholder response
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True)