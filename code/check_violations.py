import json

def check_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    vertices = data['data']['vertex']
    violations = []
    
    for key, attr in vertices.items():
        z = attr.get('z', 0)
        ub = attr.get('ub', None)
        lb = attr.get('lb', None)
        
        if ub is not None and z > ub + 1e-6:
            violations.append(f"Vertex {key}: z={z:.4f} > ub={ub:.4f} (diff={z-ub:.4f})")
        if lb is not None and z < lb - 1e-6:
            violations.append(f"Vertex {key}: z={z:.4f} < lb={lb:.4f} (diff={lb-z:.4f})")
            
    print(f"--- Results for {filename} ---")
    if not violations:
        print("No violations found.")
    else:
        print(f"Found {len(violations)} violations:")
        for v in violations[:10]:
            print(v)
        if len(violations) > 10:
            print("...")

check_json('thrust_min.json')
check_json('thrust_max.json')
