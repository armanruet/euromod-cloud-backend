from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import subprocess
import tempfile
import os
import json
import uuid
import time
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(title="EUROMOD Cloud API", version="1.0.0")

# CORS configuration for GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://armanruet.github.io",  # Your specific GitHub Pages URL
        "https://*.github.io",  # All GitHub Pages
        "http://localhost:3000",  # Local development
        "http://localhost:8000",  # Local development
        "*"  # Allow all origins for testing (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global variables for caching and job tracking
analysis_cache = {}
job_results = {}
job_status = {}


class EuromodRequest(BaseModel):
    base_system: str
    reform_system: str


class EuromodResponse(BaseModel):
    job_id: str
    status: str
    message: str


@app.get("/")
async def root():
    return {"message": "EUROMOD Cloud API is running!", "status": "healthy"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/run_euromod", response_model=EuromodResponse)
async def run_euromod_analysis(request: EuromodRequest, background_tasks: BackgroundTasks):
    try:
        # Create cache key
        cache_key = f"{request.base_system}_{request.reform_system}"

        # Check cache first
        if cache_key in analysis_cache:
            return EuromodResponse(
                job_id="cached",
                status="completed",
                message="Results retrieved from cache"
            )

        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Initialize job status
        job_status[job_id] = {
            "status": "starting",
            "progress": 0,
            "message": "Initializing EUROMOD analysis..."
        }

        # Start background task
        background_tasks.add_task(
            run_euromod_analysis_background, job_id, request.base_system, request.reform_system)

        return EuromodResponse(
            job_id=job_id,
            status="started",
            message="EUROMOD analysis started in background"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error starting EUROMOD analysis: {str(e)}")


@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    if job_id == "cached":
        return {"status": "completed", "progress": 100, "message": "Results from cache"}

    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")

    status_info = job_status[job_id]

    # If job is completed, include results
    if status_info["status"] == "completed" and job_id in job_results:
        return {
            **status_info,
            "results": job_results[job_id]
        }

    return status_info


@app.get("/cache_stats")
async def get_cache_stats():
    return {
        "cache_size": len(analysis_cache),
        "cached_analyses": list(analysis_cache.keys()),
        "active_jobs": len([j for j in job_status.values() if j["status"] in ["starting", "running"]])
    }


async def run_euromod_analysis_background(job_id: str, base_system: str, reform_system: str):
    try:
        # Update status
        job_status[job_id]["status"] = "running"
        job_status[job_id]["progress"] = 10
        job_status[job_id]["message"] = "Setting up EUROMOD environment..."

        # Get environment variables for paths
        model_path = os.getenv("EUROMOD_MODEL_PATH", "euromod_data")
        data_path = os.getenv("EUROMOD_DATA_PATH",
                              "euromod_data/Input/LU_training_data.txt")

        # Debug: Print file paths and check if files exist
        print(f"DEBUG: Model path: {model_path}")
        print(f"DEBUG: Data path: {data_path}")
        print(f"DEBUG: Data file exists: {os.path.exists(data_path)}")
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        print(f"DEBUG: Directory contents: {os.listdir('.')}")
        if os.path.exists('euromod_data'):
            print(
                f"DEBUG: euromod_data contents: {os.listdir('euromod_data')}")
            if os.path.exists('euromod_data/Input'):
                print(
                    f"DEBUG: Input contents: {os.listdir('euromod_data/Input')}")

        # Create temporary script file
        script_filename = f"temp_script_{job_id}.py"

        # Create the EUROMOD analysis script
        script_content = f'''
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings
import json
import time
from datetime import datetime
warnings.filterwarnings('ignore')

class EuromodStatisticsFixed:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.weights = data['weight'] if 'weight' in data.columns else pd.Series([1.0] * len(data))
    
    def calculate_direct_taxes(self, system: str) -> float:
        """Calculate total direct taxes for a given system"""
        try:
            # Try different possible column names
            possible_tax_cols = [
                f'{{system}}_direct_tax',
                f'{{system}}_income_tax',
                f'{{system}}_tax',
                'direct_tax',
                'income_tax'
            ]
            
            for col in possible_tax_cols:
                if col in self.data.columns:
                    result = (self.data[col] * self.weights).sum()
                    print(f"DEBUG: Found tax column {{col}} for {{system}}: {{result}}")
                    return result
            
            # If no tax column found, return 0
            print(f"DEBUG: No tax column found for {{system}}, returning 0")
            return 0.0
            
        except Exception as e:
            print(f"Error calculating direct taxes for {{system}}: {{e}}")
            return 0.0
    
    def calculate_total_household_market_incomes(self, system: str) -> float:
        """Calculate total household market incomes for a given system"""
        try:
            # Try different possible column names
            possible_income_cols = [
                f'{{system}}_market_income',
                f'{{system}}_gross_income',
                f'{{system}}_income',
                'market_income',
                'gross_income',
                'income'
            ]
            
            for col in possible_income_cols:
                if col in self.data.columns:
                    result = (self.data[col] * self.weights).sum()
                    print(f"DEBUG: Found income column {{col}} for {{system}}: {{result}}")
                    return result
            
            # If no income column found, return 0
            print(f"DEBUG: No income column found for {{system}}, returning 0")
            return 0.0
            
        except Exception as e:
            print(f"Error calculating market incomes for {{system}}: {{e}}")
            return 0.0

# Job ID for this analysis
job_id = "{job_id}"

# Progress tracking function
def update_progress(progress, message):
    import os
    script_name = os.path.basename(__file__)
    job_id_from_filename = script_name.replace('temp_script_', '').replace('.py', '')
    with open(f'progress_{{job_id_from_filename}}.json', 'w') as f:
        json.dump({{"progress": progress, "message": message}}, f)

try:
    update_progress(20, "Loading EUROMOD data...")
    
    # Load data with cloud-friendly paths
    MODEL_PATH = "{model_path}"
    data = pd.read_csv("{data_path}", sep="\\t")
    
    # Debug: Print data info
    print(f"DEBUG: Data shape: {{data.shape}}")
    print(f"DEBUG: Data columns: {{list(data.columns)}}")
    print(f"DEBUG: Data head: {{data.head()}}")
    
    update_progress(40, "Initializing EUROMOD statistics...")
    
    # Initialize statistics calculator
    stats = EuromodStatisticsFixed(data)
    
    update_progress(60, "Calculating baseline system statistics...")
    
    # Calculate baseline system statistics
    direct_taxes_baseline = stats.calculate_direct_taxes("{base_system}")
    total_household_market_incomes_baseline = stats.calculate_total_household_market_incomes("{base_system}")
    
    update_progress(80, "Calculating reform system statistics...")
    
    # Calculate reform system statistics
    direct_taxes_reform = stats.calculate_direct_taxes("{reform_system}")
    total_household_market_incomes_reform = stats.calculate_total_household_market_incomes("{reform_system}")
    
    update_progress(90, "Finalizing results...")
    
    # Prepare results
    current_time = datetime.now()
    results = {{
        'direct_taxes_baseline': float(direct_taxes_baseline),
        'direct_taxes_reform': float(direct_taxes_reform),
        'total_household_market_incomes_baseline': float(total_household_market_incomes_baseline),
        'total_household_market_incomes_reform': float(total_household_market_incomes_reform),
        'base_system': "{base_system}",
        'reform_system': "{reform_system}",
        'timestamp': current_time.isoformat(),
        'status': 'success'
    }}
    
    print(f"DEBUG: Results prepared: {{results}}")
    
    update_progress(100, "Analysis completed successfully!")
    
    # Save results
    with open(f'results_{{job_id}}.json', 'w') as f:
        json.dump(results, f)
        
except Exception as e:
    error_result = {{
        'error': str(e),
        'base_system': "{base_system}",
        'reform_system': "{reform_system}",
        'timestamp': datetime.now().isoformat(),
        'status': 'error'
    }}
    with open(f'results_{{job_id}}.json', 'w') as f:
        json.dump(error_result, f)
'''

        # Write script to temporary file
        with open(script_filename, 'w') as f:
            f.write(script_content)

        # Update status
        job_status[job_id]["progress"] = 15
        job_status[job_id]["message"] = "Executing EUROMOD analysis..."

        # Execute the script
        python_executable = os.getenv("PYTHON_EXECUTABLE", "python")
        process = await asyncio.create_subprocess_exec(
            python_executable, script_filename,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Monitor progress
        while True:
            # Check progress file
            progress_file = f"progress_{job_id}.json"
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress_data = json.load(f)
                    job_status[job_id]["progress"] = progress_data["progress"]
                    job_status[job_id]["message"] = progress_data["message"]
                except:
                    pass

            # Check if process is done
            if process.returncode is not None:
                break

            await asyncio.sleep(1)

        # Wait for process to complete
        stdout, stderr = await process.communicate()

        # Read results
        results_file = f"results_{job_id}.json"
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)

            # Cache the results
            cache_key = f"{base_system}_{reform_system}"
            analysis_cache[cache_key] = results

            # Store results
            job_results[job_id] = results

            # Update final status
            job_status[job_id]["status"] = "completed"
            job_status[job_id]["progress"] = 100
            job_status[job_id]["message"] = "Analysis completed successfully!"
        else:
            raise Exception("Results file not found")

        # Clean up temporary files
        try:
            os.remove(script_filename)
            if os.path.exists(progress_file):
                os.remove(progress_file)
            if os.path.exists(results_file):
                os.remove(results_file)
        except:
            pass

    except Exception as e:
        job_status[job_id]["status"] = "error"
        job_status[job_id]["message"] = f"Error: {str(e)}"

        # Clean up temporary files
        try:
            os.remove(script_filename)
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    # Railway sets PORT environment variable
    port = int(os.getenv("PORT", 8001))
    print(f"Starting EUROMOD Cloud API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
