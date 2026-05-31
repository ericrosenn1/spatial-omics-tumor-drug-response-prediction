# histology_response_model_v2 configs

Tracked template:

- histology_response_model_v2.example.yaml

For local runs, copy the tracked template to the runtime filename expected by the runner and edit local paths there:

```powershell
Copy-Item .\histology_response_model_v2.example.yaml .\histology_response_model_v2.yaml
```

Changing YAML values is a scientific/configuration change and should be recorded before rerunning the pipeline. Local machine-specific YAML files should not be committed.
