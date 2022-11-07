# DeepSparkHub Contributing Guidelines

You are welcome to contribute to DeepSparkHub project. We want to make contributing to this project as easy and transparent as possible.

## Contributor License Agreement

It's required to sign CLA on [Gitee](https://gitee.com/organizations/deep-spark/cla/deep-park-contributor-protocol) or [GitHub](https://gist.github.com/Deep-Spark/67e39ac5b1836a9ba5f35b1a3e600223) when you first time submit PR (Pull Request) to the DeepSparkHub project. You only need to do it once.

## Report Issue

Please submit an issue on [Gitee](https://gitee.com/deep-spark/deepsparkhub/issues) or [GitHub](https://github.com/Deep-Spark/deepsparkhub/issues) when you find a bug, or raise feature request, or raise any questions regarding this project. Please ensure your issue description is clear and sufficient enough by referring to following format:
* Whether this is a bug or a feature request or a technical question.
* The detailed description of the hardware and software environment.
* Issue description, including what happened, what you expect to happen, and how to reproduce it.
* Any special notes for your reviewers.

## Propose Pull Request (PR)

* Raise your idea as an issue on [Gitee](https://gitee.com/deep-spark/deepsparkhub/issues) or [GitHub](https://github.com/Deep-Spark/deepsparkhub/issues).
* If it is a new feature, a design document should also be submitted.
* After finishing design review and/or reaching consensus in the issue discussion, complete the code/document development on the forked repo and submit a PR on [Gitee](https://gitee.com/deep-spark/deepsparkhub/pulls) or [GitHub](https://github.com/Deep-Spark/deepsparkhub/pulls).
  * Any irrelevant changes of this PR should be avoided.
  * Make sure your commit history being ordered.
  * Always keep your branch up with the master branch.
  * Make sure all related issues being linked with this PR.
  * Need to provide test method and test result.
* You need to sign CLA on [Gitee](https://gitee.com/organizations/deep-spark/cla/deep-park-contributor-protocol) or [GitHub](https://gist.github.com/Deep-Spark/67e39ac5b1836a9ba5f35b1a3e600223) when first time submitting the PR (only do it once).
* The PR will only be merged by project maintainer after receiving 2+ LGTM from approvers. Please note that the approver is NOT allowed to add LGTM on his own PR.
* After PR is sufficiently discussed/reviewed, it will get merged, abandoned or rejected by the project maintainer.

## Contribute a New Model

* Please make sure your code is consistent with DeepSparkHub's existing coding style.
* Construct directory structure for new model as follows.
```
	# Directory Structure

	deepsparkhub
	├── cv                                    
	│   └── classification                                 
	│       ├── resnet50                           
	│       │   └── pytorch        
	│       │       ├── README.md                     
	│       │       └── train.py         
	│       ├── mobilenetv3
	│       └── ...                                 
	├── nlp                                   
	├── speech                                   
	└── 3d-reconstruction
```
                                      
* Uploaded content should contain only scripts, code, and documentation. Do not upload any data sets or directories/files generated during execution.
* The code for each model should be its own closure that can be migrated and used independently. 
* Do not include any of your personal information, such as your host IP, personal password, local directory, etc.
* Be sure to specify any additional Python libraries required with corresponding versions (if explicitly required) in the 'requirements.txt' file.
* The submitted code should be fully reviewed and self-checked and ideally passed by CI test.
* Don't forget to create a README.md file under model’s root directory, update model info by referring to the format of [Model Description Template](README_TEMPLATE.md).

## Maintenance and Communication

We appreciate your contribution to the DeepSpark community, and please keep an eye on your code after you complete a submission. You can mark your signature, email address and other contact information in the README of the submitted model.
Other developers may be using the model you submitted and may have some questions during use. In this case, you can communicate with them in detail through issues, in-site messages, emails, etc.

## Coding Style

The Python coding style suggested by [Python PEP 8 Coding Style](https://www.python.org/dev/peps/pep-0008/) and C++ coding style suggested by [Google C++ Coding Guidelines](http://google.github.io/styleguide/cppguide.html) are used in DeepSpark community.

## License

By contributing to DeepSparkHub project, you agree that your contribution will follow the license under the LICENSE file in the root directory of the source tree.
