from defog.util import identify_categorical_columns
from defog import Defog

class KpaasGenerateSchema(Defog):
    
    def aa(self, message: str) -> str:
        return f"MyService: {message}"

    # 重新定义mysql实现方式
    def generate_cloumn_mysql_schema(
        self,
        tables: list,
        upload: bool = True,
        return_format: str = "csv",
        scan: bool = True,
        return_tables_only: bool = False,
    ) -> str:
        try:
            import mysql.connector
        except:
            raise Exception("mysql-connector not installed.")

        conn = mysql.connector.connect(**self.db_creds)
        cur = conn.cursor()
        schemas = {}

        if len(tables) == 0:
            # get all tables
            db_name = self.db_creds.get("database", "")
            cur.execute(
                f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{db_name}';"
            )
            tables = [row[0] for row in cur.fetchall()]

        if return_tables_only:
            conn.close()
            return tables

        print("Getting schema for the relevant table in your database...")
        # get the schema for each table
        for table_name in tables:
            schema_name = self.db_creds.get("database", "")
            cur.execute(
                """
                SELECT columns.column_name, columns.data_type,columns.column_comment,TABLES.TABLE_COMMENT 
                    FROM information_schema.columns		
                        JOIN information_schema.TABLES 	
                        on columns.TABLE_NAME = TABLES.TABLE_NAME	
                        AND columns.TABLE_SCHEMA = TABLES.TABLE_SCHEMA 
                        WHERE 
                        columns.TABLE_SCHEMA = %s
                        and columns.table_name = %s;
                """,
                (schema_name,table_name),
            )
            rows = cur.fetchall()
            rows = [row for row in rows]
            rows = [{"column_name": i[0], "data_type": i[1],"column_description":i[2],"table_description":i[3]} for i in rows]
            print(f"==================rows=={table_name}start===============")
            print(rows)
            print(f"==================rows=={table_name}end===============")
            if scan:
                rows = identify_categorical_columns(cur, table_name, rows)
            
            print(f"==================rows==after scan===============")
            print(rows)
            if len(rows) > 0:
                schemas[table_name] = rows
        conn.close()

        if upload:
            r = requests.post(
                f"{self.base_url}/get_schema_csv",
                json={
                    "api_key": self.api_key,
                    "schemas": schemas,
                    "foreign_keys": [],
                    "indexes": [],
                },
                verify=False,
            )
            resp = r.json()
            if "csv" in resp:
                csv = resp["csv"]
                if return_format == "csv":
                    pd.read_csv(StringIO(csv)).to_csv("defog_metadata.csv", index=False)
                    return "defog_metadata.csv"
                else:
                    return csv
            else:
                print(f"We got an error!")
                if "message" in resp:
                    print(f"Error message: {resp['message']}")
                print(
                    f"Please feel free to open a github issue at https://github.com/defog-ai/defog-python if this a generic library issue, or email support@defog.ai."
                )
        else:
            return schemas


   # 重新定义mysql实现方式
    def generate_table_mysql_schema(
        self,
        tables: list,
        upload: bool = True,
        return_format: str = "csv",
        scan: bool = True,
        return_tables_only: bool = False,
    ) -> str:
        try:
            import mysql.connector
        except:
            raise Exception("mysql-connector not installed.")

        conn = mysql.connector.connect(**self.db_creds)
        cur = conn.cursor()
        schemas = {}

        if len(tables) == 0:
            # get all tables
            db_name = self.db_creds.get("database", "")
            cur.execute(
                f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{db_name}';"
            )
            tables = [row[0] for row in cur.fetchall()]

        if return_tables_only:
            conn.close()
            return tables

        print("Getting schema for the relevant table in your database...")
        # get the schema for each table
        for table_name in tables:
            schema_name = self.db_creds.get("database", "")
            cur.execute(
                """
                 SELECT TABLES.TABLE_NAME,  TABLES.TABLE_COMMENT 
                    FROM information_schema.TABLES 	
                        WHERE 
                        TABLES.TABLE_SCHEMA = %s
                        and TABLES.table_name = %s;
                """,
                (schema_name,table_name),
            )
            rows = cur.fetchall()
            rows = [row for row in rows]
            rows = [{"table_description":i[1]} for i in rows]
            print(f"==================rows=={table_name}start===============")
            print(rows)
            print(f"==================rows=={table_name}end===============")
            # if scan:
            #     rows = identify_categorical_columns(cur, table_name, rows)
            
            # print(f"==================rows==after scan===============")
            print(rows)
            if len(rows) > 0:
                schemas[table_name] = rows
        conn.close()

        if upload:
            r = requests.post(
                f"{self.base_url}/get_schema_csv",
                json={
                    "api_key": self.api_key,
                    "schemas": schemas,
                    "foreign_keys": [],
                    "indexes": [],
                },
                verify=False,
            )
            resp = r.json()
            if "csv" in resp:
                csv = resp["csv"]
                if return_format == "csv":
                    pd.read_csv(StringIO(csv)).to_csv("defog_metadata.csv", index=False)
                    return "defog_metadata.csv"
                else:
                    return csv
            else:
                print(f"We got an error!")
                if "message" in resp:
                    print(f"Error message: {resp['message']}")
                print(
                    f"Please feel free to open a github issue at https://github.com/defog-ai/defog-python if this a generic library issue, or email support@defog.ai."
                )
        else:
            return schemas

