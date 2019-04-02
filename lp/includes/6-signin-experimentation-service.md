As previously mentioned,  you use the code shown below to access your workspace. Log in with your Azure account if required. Replace name, subscription_id and resource_group parameters with those of your Workspace.

```python
# import package and use get function to access Workspace
from azureml.core import Workspace,Experiment ,Run
from azureml.core import Workspace
ws = Workspace.get(name='course_trial',
                      subscription_id='61f7cffa-e418-4a80-9679-ef35724532a8', 
                      resource_group='docsaml'
                     )
```