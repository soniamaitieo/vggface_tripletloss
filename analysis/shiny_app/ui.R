library(shiny)

columnsList <- list("Attractive", 
                    "Feminine",
                    "dist_centre_F", 
                    "dist_centre_H",
                    "LL",
                    "LL_F",
                    "LL_M",
                    "proba_F_svm",
                    "pF_svm_linear")

# Define UI for application that draws a histogram
ui <- shinyUI(fluidPage(
  
  # Application title
  titlePanel("CFD Analysis"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
      fileInput("file", h3("File input")),
      hr(),
      selectInput("xaxis", 
                  label = "Choose X axis column",
                  choices = columnsList,
                  selected = "Attractive"),
      
      h3("Ranked"),
      checkboxInput("xrank", "Rank X Axis", value = FALSE),
      
      hr(),
      selectInput("yaxis",
                  label = "Choose Y axis column",
                  choices = columnsList,
                  selected = "LL"),
      
      h3("Ranked"),
      checkboxInput("yrank", "Rank Y Axis", value = FALSE),
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
      plotOutput("distPlot")
    )
  )
))