#!/usr/bin/env python3
"""
Script to extract PostgreSQL schema information from demo_bank database
and build a FAISS index for retrieval.

Usage:
    python src/data_prep/build_schema_index.py [--api-key API_KEY] [--output-dir OUTPUT_DIR]
    # or as a module:
    python -m src.data_prep.build_schema_index [--api-key API_KEY] [--output-dir OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import psycopg2
from psycopg2.extras import RealDictCursor
import faiss
import numpy as np
import requests
from src.config import EMBEDDING_MODEL, EMBEDDING_BASE_URL

load_dotenv()

def get_db_connection():
    """Create a PostgreSQL connection using environment variables."""
    conn_params = {
        'host': os.getenv('PGHOST', 'localhost'),
        'port': os.getenv('PGPORT', '5432'),
        'database': os.getenv('PGDATABASE', 'demo_bank'),
        'user': os.getenv('PGUSER', 'postgres'),
        'password': os.getenv('PGPASSWORD', ''),
    }
    
    try:
        conn = psycopg2.connect(**conn_params)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)


def get_table_info(conn) -> List[Dict[str, Any]]:
    """Extract comprehensive table information from the database."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Get all tables in the public schema
        cur.execute("""
            SELECT 
                table_name,
                table_type
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)
        tables = cur.fetchall()
        
        table_info = []
        for table in tables:
            table_name = table['table_name']
            
            # Get column information
            cur.execute("""
                SELECT 
                    column_name,
                    data_type,
                    character_maximum_length,
                    is_nullable,
                    column_default,
                    ordinal_position
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = %s
                ORDER BY ordinal_position;
            """, (table_name,))
            columns = cur.fetchall()
            
            # Get primary key information
            cur.execute("""
                SELECT 
                    kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_schema = 'public'
                AND tc.table_name = %s
                ORDER BY kcu.ordinal_position;
            """, (table_name,))
            primary_keys = [row['column_name'] for row in cur.fetchall()]
            
            # Get foreign key information
            cur.execute("""
                SELECT
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name,
                    tc.constraint_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = 'public'
                AND tc.table_name = %s;
            """, (table_name,))
            foreign_keys = cur.fetchall()
            
            # Get unique constraints
            cur.execute("""
                SELECT
                    kcu.column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'UNIQUE'
                AND tc.table_schema = 'public'
                AND tc.table_name = %s;
            """, (table_name,))
            unique_columns = [row['column_name'] for row in cur.fetchall()]
            
            # Get check constraints
            cur.execute("""
                SELECT
                    cc.constraint_name,
                    cc.check_clause
                FROM information_schema.check_constraints cc
                JOIN information_schema.constraint_column_usage ccu
                    ON cc.constraint_name = ccu.constraint_name
                WHERE ccu.table_schema = 'public'
                AND ccu.table_name = %s;
            """, (table_name,))
            check_constraints = cur.fetchall()
            
            # Get indexes
            cur.execute("""
                SELECT
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                AND tablename = %s;
            """, (table_name,))
            indexes = cur.fetchall()
            
            # Get row count
            cur.execute(f"SELECT COUNT(*) as row_count FROM {table_name};")
            row_count = cur.fetchone()['row_count']
            
            table_info.append({
                'table_name': table_name,
                'columns': [
                    {
                        'name': col['column_name'],
                        'data_type': col['data_type'],
                        'max_length': col['character_maximum_length'],
                        'nullable': col['is_nullable'] == 'YES',
                        'default': col['column_default'],
                        'position': col['ordinal_position']
                    }
                    for col in columns
                ],
                'primary_keys': primary_keys,
                'foreign_keys': [
                    {
                        'column': fk['column_name'],
                        'references_table': fk['foreign_table_name'],
                        'references_column': fk['foreign_column_name'],
                        'constraint_name': fk['constraint_name']
                    }
                    for fk in foreign_keys
                ],
                'unique_columns': unique_columns,
                'check_constraints': [
                    {
                        'name': cc['constraint_name'],
                        'clause': cc['check_clause']
                    }
                    for cc in check_constraints
                ],
                'indexes': [
                    {
                        'name': idx['indexname'],
                        'definition': idx['indexdef']
                    }
                    for idx in indexes
                ],
                'row_count': row_count
            })
        
        return table_info


def format_schema_chunks(table_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format schema information into chunks suitable for embedding."""
    chunks = []
    
    for table in table_info:
        # Chunk 1: Table overview
        table_overview = f"Table: {table['table_name']}\n"
        table_overview += f"Row count: {table['row_count']}\n\n"
        
        if table['primary_keys']:
            table_overview += f"Primary key(s): {', '.join(table['primary_keys'])}\n"
        
        if table['unique_columns']:
            table_overview += f"Unique column(s): {', '.join(table['unique_columns'])}\n"
        
        chunks.append({
            'type': 'table_overview',
            'table_name': table['table_name'],
            'content': table_overview.strip(),
            'metadata': {
                'row_count': table['row_count'],
                'primary_keys': table['primary_keys'],
                'unique_columns': table['unique_columns']
            }
        })
        
        # Chunk 2: Column details
        column_details = f"Table: {table['table_name']}\nColumns:\n"
        for col in table['columns']:
            col_desc = f"  - {col['name']}: {col['data_type']}"
            if col['max_length']:
                col_desc += f"({col['max_length']})"
            if not col['nullable']:
                col_desc += " NOT NULL"
            if col['default']:
                col_desc += f" DEFAULT {col['default']}"
            column_details += col_desc + "\n"
        
        chunks.append({
            'type': 'columns',
            'table_name': table['table_name'],
            'content': column_details.strip(),
            'metadata': {
                'columns': [
                    {
                        'name': col['name'],
                        'data_type': col['data_type'],
                        'nullable': col['nullable']
                    }
                    for col in table['columns']
                ]
            }
        })
        
        # Chunk 3: Foreign key relationships
        if table['foreign_keys']:
            fk_details = f"Table: {table['table_name']}\nForeign Key Relationships:\n"
            for fk in table['foreign_keys']:
                fk_details += f"  - {fk['column']} -> {fk['references_table']}.{fk['references_column']}\n"
            
            chunks.append({
                'type': 'foreign_keys',
                'table_name': table['table_name'],
                'content': fk_details.strip(),
                'metadata': {
                    'foreign_keys': table['foreign_keys']
                }
            })
        
        # Chunk 4: Constraints and indexes
        if table['check_constraints'] or table['indexes']:
            constraints_details = f"Table: {table['table_name']}\n"
            
            if table['check_constraints']:
                constraints_details += "Check Constraints:\n"
                for cc in table['check_constraints']:
                    constraints_details += f"  - {cc['name']}: {cc['clause']}\n"
            
            if table['indexes']:
                constraints_details += "Indexes:\n"
                for idx in table['indexes']:
                    constraints_details += f"  - {idx['name']}: {idx['definition']}\n"
            
            chunks.append({
                'type': 'constraints_indexes',
                'table_name': table['table_name'],
                'content': constraints_details.strip(),
                'metadata': {
                    'check_constraints': table['check_constraints'],
                    'indexes': table['indexes']
                }
            })
        
        # Chunk 5: Cross-table relationships (for each foreign key)
        for fk in table['foreign_keys']:
            relationship = (
                f"Relationship: {table['table_name']}.{fk['column']} "
                f"references {fk['references_table']}.{fk['references_column']}"
            )
            chunks.append({
                'type': 'relationship',
                'table_name': table['table_name'],
                'content': relationship,
                'metadata': {
                    'from_table': table['table_name'],
                    'from_column': fk['column'],
                    'to_table': fk['references_table'],
                    'to_column': fk['references_column']
                }
            })
    
    return chunks


def get_embeddings(texts: List[str], api_key: str) -> np.ndarray:
    """Generate embeddings for a list of texts using Fireworks API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    embeddings = []
    # Process in batches to avoid rate limits
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            # Fireworks API expects a list of inputs
            payload = {
                "model": EMBEDDING_MODEL,
                "input": batch
            }
            response = requests.post(EMBEDDING_BASE_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            batch_embeddings = [item["embedding"] for item in result["data"]]
            embeddings.extend(batch_embeddings)
            print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        except Exception as e:
            print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            raise
    
    return np.array(embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray, output_dir: Path):
    """Build and save a FAISS index."""
    # Get embedding dimension
    dimension = embeddings.shape[1]
    
    # Create FAISS index (using L2 distance - Inner Product is also common)
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to index
    index.add(embeddings)
    
    # Save index as binary file
    index_path = output_dir / 'schema_index.bin'
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}")
    print(f"Index contains {index.ntotal} vectors with dimension {dimension}")
    
    return index


def save_embeddings(embeddings: np.ndarray, output_dir: Path):
    """Save embeddings as a NumPy .npy file."""
    embeddings_path = output_dir / 'embeddings.npy'
    np.save(str(embeddings_path), embeddings)
    print(f"Saved embeddings to {embeddings_path}")
    print(f"Embeddings shape: {embeddings.shape} (num_vectors, dimension)")


def main():
    parser = argparse.ArgumentParser(
        description='Build FAISS index from PostgreSQL schema information'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.getenv('FIREWORKS_API_KEY', ''),
        help='Fireworks API key (or set FIREWORKS_API_KEY env var)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for index and metadata (default: data)'
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: API key required. Set FIREWORKS_API_KEY env var or use --api-key")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Connecting to PostgreSQL database...")
    conn = get_db_connection()
    print("Connected successfully!")
    
    print("\nExtracting schema information...")
    table_info = get_table_info(conn)
    conn.close()
    print(f"Found {len(table_info)} tables")
    
    print("\nFormatting schema into chunks...")
    chunks = format_schema_chunks(table_info)
    print(f"Created {len(chunks)} chunks")
    
    # Save raw chunks metadata
    chunks_metadata_path = output_dir / 'chunks_metadata.json'
    with open(chunks_metadata_path, 'w') as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved chunks metadata to {chunks_metadata_path}")
    
    print("\nGenerating embeddings...")
    texts = [chunk['content'] for chunk in chunks]
    embeddings = get_embeddings(texts, args.api_key)
    print(f"Generated {len(embeddings)} embeddings")
    
    print("\nSaving embeddings...")
    save_embeddings(embeddings, output_dir)
    
    print("\nBuilding FAISS index...")
    index = build_faiss_index(embeddings, output_dir)
    
    # Save table info for reference
    table_info_path = output_dir / 'table_info.json'
    with open(table_info_path, 'w') as f:
        json.dump(table_info, f, indent=2)
    print(f"Saved table info to {table_info_path}")
    
    print("\n✅ Schema index built successfully!")
    print(f"\nFiles created in {output_dir}:")
    print(f"  - embeddings.npy (embeddings array)")
    print(f"  - schema_index.bin (FAISS index)")
    print(f"  - chunks_metadata.json (chunk metadata for retrieval)")
    print(f"  - table_info.json (full table information)")


if __name__ == '__main__':
    main()

