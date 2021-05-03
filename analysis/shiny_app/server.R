source("analyses_fn.R")
library(shiny)

# Define server logic
server <- shinyServer(function(input, output) {

  
  output$distPlot <- renderPlot({
    
    inFile <- input$file
    
    if (is.null(inFile)) {
      # Default analysis file
      # TODO: Get file with relative path
      CFD <- read.csv("/home/sonia/vggface_tripletloss/analysis/shiny_app/data/CFD_N_analysis.csv", row.names = 1)
    } else {
      CFD <- read.csv(inFile$file, row.names=1)
    }
    
    xvalue <- input$xaxis
    if(input$xrank) {
      xvalue <- paste("rank(", input$xaxis, ")")
    }
    yvalue <- input$yaxis
    if(input$yrank) {
      yvalue <- paste("rank(", input$yaxis, ")")
    }
    
    makeggplot(CFD, xvalue, yvalue, "GenderSelf", stat_cor())
  })
  
  output$acpPlot <- renderPlot({
    
    inFile <- input$file
    
    if (is.null(inFile)) {
      # Default analysis file
      # TODO: Get file with relative path
      CFD <- read.csv("/home/soniamai/Bureau/shinyapptest/test/data/CFD_N_analysis.csv", row.names = 1)
    } else {
      CFD <- read.csv(inFile$file, row.names=1)
    }
    
    makepca(CFD)
  })
})