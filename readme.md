## Getting Started
### Creating your first RAG chain application
```
./rag ingest-data -e dev
./rag create-rag-version -e dev
```

### Evaluating and reviewing the RAG application in DEV environment
```
./rag run-online-eval -e dev
./rag run-offline-eval -e dev -v 2 -t mydataset
./rag start-review -e dev -v 2 -t someTable
```
### Set up the REVIEWER and END_USER environment
```
./rag setup-prod-env
``` 

### Promoting to REVIEWER environment 
```
# User configures the model version and eval_table_name in the bundle used for the reviewer environment
./rag deploy-chain -e reviewer -version 2
./rag run-online-eval -e reviewer
./rag run-offline-eval -e reviewer -v 2 -t mydataset
./rag start-review -e reviewer -v 2 -t someTable
```

### Promoting to END_USER environment 
```
# User configures the model version in the bundle used for the end_user environment
./rag deploy-chain -e end_user -v 2
./rag run-online-eval -e end_user
./rag run-offline-eval -e end_user -v 2 -t mydataset
./rag start-review -e end_user -v 2 -t someTable
```

## RAG bundle Structure 
```
├── databricks.yml  <-- top level configurations for the bundle 
└── resources          <-- contains resource definitions for the RAG app organized by environment
    ├── shared.yml     <-- shared bundle resources across all environments
    ├── dev.yml        <-- dev environment specific resources
    ├── reviewer.yml   <-- reviewer environment specific resources
    ├── end_user.yml   <-- end_user environment specific resources
└── src             <-- source code for all components of the RAG bundle              
    └── my_rag_builder      <-- customizable source code to build the RAG app
        └── ...
    └── eval                <-- evaluation source code
        └── ...
    └── notebooks           <-- all the notebooks used in the bundle.
        └── ...
    └── review              <-- feedback and review source dode
        └── ...
    └── utils               <-- utilities used in the bundle
        └── ...
```