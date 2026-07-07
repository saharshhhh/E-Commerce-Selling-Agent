import pandas as pd
from data_loader import load_orders
from typing import Optional, Dict

class OrderLookup:
    def __init__(self, csv_path: str):
        self.df = load_orders(csv_path)

    def get_order_details(self, order_id: str) -> Optional[Dict]:
        """Retrieve order details by order_id"""
        order = self.df[self.df['order_id'] == order_id]
        if not order.empty:
            return order.iloc[0].to_dict()
        return None

if __name__ == "__main__":
    lookup = OrderLookup("sample_data/orders.csv")
    print(lookup.get_order_details("ORD1001"))
    print(lookup.get_order_details("INVALID"))
