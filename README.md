## Analysis

# Shiny app
Using RStudio:
Use "run app" on app.R file

Using docker:
R Version 3.6.3
Create the docker image from folder analysis
```console
foo@bar:~/vggface_tripletloss/analysis$ docker build -t analysis/shiny .
```

Run the created docker image
```console
foo@bar:~/$ docker run --user shiny --rm -p 8080:3838 analysis/shiny
```

Now it can be accessed in browser on http://localhost:8080