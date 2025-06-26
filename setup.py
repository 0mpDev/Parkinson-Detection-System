import os
import shutil
import sys

def setup_project():
    print("Setting up Parkinson's Disease Detection project...")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(current_dir, 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        print(f"Created templates directory at: {templates_dir}")
    
    # Check if index.html exists in current directory
    index_src = os.path.join(current_dir, 'index.html')
    index_dest = os.path.join(templates_dir, 'index.html')
    
    if os.path.exists(index_src):
        # Copy index.html to templates directory
        shutil.copy(index_src, index_dest)
        print(f"Copied index.html to templates directory")
    else:
        print("Warning: index.html not found in current directory")
    
    # Check if model files exist
    model_files = ['model.pkl', 'scaler.pkl', 'selector.pkl', 'pca.pkl']
    missing_files = []
    
    for file in model_files:
        if not os.path.exists(os.path.join(current_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print("\nWARNING: The following model files are missing:")
        for file in missing_files:
            print(f" - {file}")
        print("\nYou need to run main.py first to generate these files.")
    else:
        print("\nAll model files found. The application is ready to run.")
    
    print("\nSetup completed.")
    print("\nTo run the application, use the command:")
    print("python app.py")

if __name__ == "__main__":
    setup_project()