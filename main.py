import argparse
import pandas as pd
import uvicorn
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retention import RetentionEngine

app = FastAPI(title="Retention API")
engine = RetentionEngine()
CACHE = {}

class OptimizeReq(BaseModel):
    budget: float
    cost: float

@app.on_event("startup")
def startup():
    print("Loading data...")
    # Lazy load to allow fast startup, full training triggers on first request or manually
    df = engine.load_data()
    CACHE['df'] = df
    print("Data loaded. Ready.")

@app.post("/predict")
def predict(req: OptimizeReq):
    if 'df' not in CACHE: raise HTTPException(503, "Data not loaded")
    
    # Just-in-Time Training
    if not engine.is_fitted:
        print("Training model...")
        engine.train(CACHE['df'], tune_hyperparams=False)
    
    # Run Optimization
    plan = engine.optimize_budget(CACHE['df'], req.budget, req.cost)
    
    # Save request log
    if not plan.empty:
        # Create a descriptive filename with inputs and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"api_req_b{int(req.budget)}_c{int(req.cost)}_{timestamp}.csv"
        save_path = f"logs/{filename}"
        
        plan.to_csv(save_path, index=False)
        print(f"Log saved: {save_path}")

    return {
        "status": "success",
        "meta": {
            "budget": req.budget,
            "cost": req.cost,
            "audit_file": filename if not plan.empty else None
        },
        "selected_count": len(plan),
        "total_spend": len(plan) * req.cost,
        "protected_value": plan['value_at_risk'].sum() if not plan.empty else 0,
        "plan": plan.to_dict(orient="records")
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retention Optimizer CLI")
    parser.add_argument("mode", choices=["run", "serve"], help="Execution Mode")
    parser.add_argument("--budget", type=float, default=5000, help="Marketing Budget ($)")
    parser.add_argument("--cost", type=float, default=20, help="Cost per Intervention ($)")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip hyperparameter search (Faster)")
    
    args = parser.parse_args()

    if args.mode == "run":
        print(f"\nStarting batch process...")
        print(f"   Budget: ${args.budget} | Cost/Action: ${args.cost}")
        
        engine = RetentionEngine()
        df = engine.load_data()
        
        # Train with full comparison unless skipped
        engine.train(df, tune_hyperparams=not args.skip_tuning)
        
        # Optimize
        plan = engine.optimize_budget(df, args.budget, args.cost)
        
        print("\nDone.")
        if not plan.empty:
            print(f"   Selected Customers: {len(plan)}")
            print(f"   Budget Utilized:    ${len(plan)*args.cost}")
            print(f"   Value Protected:    ${plan['value_at_risk'].sum():,.2f}")
            print(f"   Avg ROI:            {plan['roi'].mean():.2f}x")
            
            out_path = "logs/final_intervention_plan.csv"
            plan.to_csv(out_path, index=False)
            print(f"\nPlan saved to: {out_path}")
            print(f"Assets generated in: assets/")
        else:
            print("No customers met ROI criteria.")

    elif args.mode == "serve":
        print("Starting API server on port 8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)