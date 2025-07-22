#!/bin/env python3

#get rid of sagemaker spam
import logging
sagemaker_config_logger = logging.getLogger("sagemaker.config")
sagemaker_config_logger.setLevel(logging.WARNING)

import sagemaker



sagemaker_session = sagemaker.Session()
#role = sagemaker.get_execution_role()

bucket = sagemaker_session.default_bucket()

print(bucket)