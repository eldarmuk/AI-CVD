"""
Utility functions for querying and exporting data from the database.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from database import get_connection

DB_PATH = Path(__file__).parent.parent / "db" / "hrp_data.db"


class DataStore:
    """Wrapper for convenient data access."""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = None
    
    def __enter__(self):
        self.conn = get_connection(self.db_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def query(self, sql: str, **params) -> pd.DataFrame:
        """Execute SQL query and return as DataFrame."""
        return pd.read_sql(sql, self.conn, params=params)
    
    def get_measurements_by_type(self, measurement_type: str,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get all measurements for a specific type.
        
        Args:
            measurement_type: Type name (e.g., 'Heart Rate', 'Steps')
            start_date: Optional start date (ISO format)
            end_date: Optional end date (ISO format)
        """
        sql = "SELECT * FROM measurements WHERE type = :mtype"
        params = {"mtype": measurement_type}
        
        if start_date:
            sql += " AND date >= :start"
            params["start"] = start_date
        
        if end_date:
            sql += " AND date <= :end"
            params["end"] = end_date
        
        sql += " ORDER BY date"
        return self.query(sql, **params)
    
    def get_senior_measurements(self, senior_id: int,
                                measurement_type: Optional[str] = None,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get all measurements for a specific senior.
        
        Args:
            senior_id: The senior ID
            measurement_type: Optional filter (e.g., 'Heart Rate', 'Steps')
            start_date: Optional start date (ISO format)
            end_date: Optional end date (ISO format)
        """
        sql = "SELECT * FROM measurements WHERE senior_id = :sid"
        params = {"sid": senior_id}
        
        if measurement_type:
            sql += " AND type = :mtype"
            params["mtype"] = measurement_type
        
        if start_date:
            sql += " AND date >= :start"
            params["start"] = start_date
        
        if end_date:
            sql += " AND date <= :end"
            params["end"] = end_date
        
        sql += " ORDER BY date"
        return self.query(sql, **params)
    
    def get_measurements_by_date_range(self, measurement_type: str,
                                       start_date: str,
                                       end_date: str) -> pd.DataFrame:
        """Get measurements for a type within a date range."""
        sql = """
            SELECT * FROM measurements
            WHERE type = :mtype
            AND date BETWEEN :start AND :end
            ORDER BY date
        """
        return self.query(sql, mtype=measurement_type, start=start_date, end=end_date)
    
    def get_medical_info(self, senior_id: int) -> pd.DataFrame:
        """Get medical information for a senior."""
        sql = "SELECT * FROM medical_info WHERE senior_id = :senior_id"
        return self.query(sql, senior_id=senior_id)
    
    def get_alerts(self, senior_id: Optional[int] = None) -> pd.DataFrame:
        """Get alerts, optionally filtered by senior."""
        if senior_id:
            sql = "SELECT * FROM alerts WHERE senior_id = :senior_id ORDER BY alert_date DESC"
            return self.query(sql, senior_id=senior_id)
        else:
            return self.query("SELECT * FROM alerts ORDER BY alert_date DESC")
    
    def get_seniors_with_diseases(self, disease_name: str) -> pd.DataFrame:
        """Find seniors with specific disease."""
        sql = """
        SELECT DISTINCT senior_id, disease_names 
        FROM medical_info 
        WHERE disease_names LIKE :disease
        """
        return self.query(sql, disease=f"%{disease_name}%")
    
    def export_to_csv(self, table_name: str, output_path: Path):
        """Export entire table to CSV."""
        df = self.query(f"SELECT * FROM {table_name}")
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} rows to {output_path}")
