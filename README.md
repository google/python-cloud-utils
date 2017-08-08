# Multi Cloud Utils

Multi Cloud Utils is a library to handle standard operations across mutliple cloud platforms.

Currently supported clouds are GCP, and AWS.

This is not an official Google product

supported opertations:

## list_instances

Example: list_instances.py foo\*bar\*ex\*
```
Cloud                      Name                            Zone                 Id                  State        Ip Address          Type             Created               Autoscaling Group          Iam Or Service Account   Private Ip Address       Project      
 aws               aws-foo-bar-exampl                   ap-west-1a      i-123456789abcdefgh        stopped          None             r3.8xl      2017-06-01 10:48:56                None                   foobar                       None                aws        
 gcp               gcp-foo-bar-exampl                   us-west1-b      0123456789012345678        running          None             mem-2       2017-06-01 10:48:56                None                   foobar                       None                gcp        
```

Recommended alias:
```alias li='path/to/list_instances.py'```
