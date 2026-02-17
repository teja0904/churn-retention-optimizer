import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pydantic import ValidationError
from retention import RetentionEngine, TransactionSchema

@pytest.fixture
def mock_transaction_data():
    end_date = datetime(2023, 12, 31)
    start_date = datetime(2023, 1, 1)
    days_range = (end_date - start_date).days
    
    n_rows = 1000
    n_customers = 50
    
    customer_ids = [str(i) for i in range(1000, 1000 + n_customers)]
    
    random_days = np.random.randint(0, days_range, n_rows)
    dates = [start_date + timedelta(days=int(d)) for d in random_days]
    
    data = {
        'Invoice': [f'INV{i}' for i in range(n_rows)],
        'StockCode': ['ITEM_A'] * n_rows,
        'Quantity': np.random.randint(1, 5, n_rows),
        'InvoiceDate': dates,
        'Price': np.random.uniform(10, 50, n_rows),
        'Customer ID': np.random.choice(customer_ids, n_rows)
    }
    df = pd.DataFrame(data)
    df['TotalValue'] = df['Quantity'] * df['Price']
    
    df.loc[0, 'Customer ID'] = '1000'
    df.loc[0, 'InvoiceDate'] = end_date - timedelta(days=1)
    
    df.loc[1, 'Customer ID'] = '1001'
    df.loc[1, 'InvoiceDate'] = start_date
    
    return df

def test_schema_rejects_negative_price():
    with pytest.raises(ValidationError):
        TransactionSchema(
            Invoice="1", StockCode="A", Quantity=1, 
            InvoiceDate=datetime.now(), Price=-10.00, CustomerID="123"
        )

def test_schema_accepts_valid_data():
    try:
        TransactionSchema(
            Invoice="1", StockCode="A", Quantity=1, 
            InvoiceDate=datetime.now(), Price=10.00, CustomerID="123"
        )
    except ValidationError:
        pytest.fail("Schema rejected valid data.")

def test_feature_engineering_integrity(mock_transaction_data):
    engine = RetentionEngine(obs_window=30)
    
    # Run pipeline step explicitly
    from lifetimes.utils import summary_data_from_transaction_data
    summary = summary_data_from_transaction_data(
        mock_transaction_data, 'Customer ID', 'InvoiceDate', monetary_value_col='TotalValue'
    )
    
    enriched = engine._enrich_features(summary.rename(columns={
        'frequency': 'frequency_cal', 'recency': 'recency_cal', 
        'T': 'T_cal', 'monetary_value': 'monetary_value_cal'
    }))
    
    assert 'velocity' in enriched.columns
    assert 'avg_interpurchase_days' in enriched.columns
    assert (enriched['velocity'] >= 0).all()

def test_model_training_execution(mock_transaction_data):
    engine = RetentionEngine(obs_window=30)
    
    summary = engine.train(mock_transaction_data, tune_hyperparams=False)
    
    assert engine.is_fitted
    assert engine.risk_model is not None
    assert not summary.empty
    assert 'churned' in summary.columns

def test_optimization_budget_strictness(mock_transaction_data):
    engine = RetentionEngine(obs_window=30)
    engine.train(mock_transaction_data, tune_hyperparams=False)
    
    budget = 50.0
    cost_per_action = 20.0
    
    plan = engine.optimize_budget(mock_transaction_data, budget, cost_per_action)
    
    if not plan.empty:
        total_spend = plan['cost'].sum()
        assert total_spend <= budget
        assert len(plan) <= 2

def test_optimization_roi_sanity(mock_transaction_data):
    engine = RetentionEngine(obs_window=30)
    engine.train(mock_transaction_data, tune_hyperparams=False)
    
    budget = 10000
    cost = 100
    
    plan = engine.optimize_budget(mock_transaction_data, budget, cost)
    
    if not plan.empty:
        assert (plan['value_at_risk'] > cost).all()
        assert (plan['roi'] > 1.0).all()