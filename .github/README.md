Boilerplate-repository
======

Simple repository/folder-structure to be used as base when crating new GitHub repositories.

### Motivation
As time goes by, the more I use GitHub. Due to this, I felt the need to create a simple boilerplate for my repositories on GitHub.

The purpose is to unify my repositories, using a folder structure, files and conventions that I find appropriate.


---


Markdown files
------

All github recommended community standards files should be within __.github__ folder, except the _LICENSE.txt_ file (this one should be in the root). Those files are:

* README.md
* CONTRIBUTING.md 
* ISSUE_TEMPLATE.md
* PULL_REQUEST_TEMPLATE.md

Also, for simplicity, there's two more files to help to write down simple tasks and register minor updates:

* TODO.md
* CHANGELOG.md

GitHub automatically search for those markdown files in the __root__ folder, __.github__ folder and the __docs__ folder. For organization purposes, the __docs__ folder will be used to keep documentation about the project. 

### Headers

The headers are mostly used in the _README.md_ file and, to provide a better structure, there's some conventions I use:

|Header|How it's done|
|:-----|:------------|
|The repository name (title) should always be a h1 written with underline-ish style.|<pre>'repo-name'<br>======</pre>|
|The sections should follow the same rule, but must be a h2 header.|<pre>'section-name'<br>------</pre>|
|The subsections headers starts at h3 and should never be as big as the title or the sections.|<pre>### 'subsection-name'</pre>|
|Between the title and every section, there will be a horizontal rule to emphasize the division.|<pre>---</pre>|

### README

For every project, the _README.md_ file must have at least the following basic headers structure:

<pre>
Repository-name (title)
======


---


Getting Started
------
  
Installation and Usage
------

### Pre requisites

### Running


---


License
------
</pre>


---


Documentation
------

All the documentation should be inside _/docs_, and can be done with _.md_ or _.html_ files. Since GitHub-page is a powerfull resource and any repository can have a web page, the documentation is usually done with _.html_ files using the [Skeleton Boilerplate](http://getskeleton.com/), and the repo link is placed in the description of the repository.


---


Folder structure
------

The main folder structure should be the same as this repository:
```
.
├── LICENSE.txt
├── .github
│   ├── README.md
│   ├── CONTRIBUTING.md 
│   ├── ISSUE_TEMPLATE.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── TODO.md
│   └── CHANGELOG.md
├── docs
│   ├── index.html
│   └── assets
│       ├── css
│       ├── js
│       └── images
└── src
    └── ALL SOURCE FILES GOES HERE
```

In need of more folders to organize the repo, do as you wish, but at least preserve this structure.
