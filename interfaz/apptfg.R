library(shiny)
library(keras)
library(reticulate)
library(imager)
ui <- fluidPage(
  
  titlePanel("prueba tfg esqueleto página", windowTitle = "TFG"),
  sidebarPanel(
    fileInput("file", "Load the image", accept = c('image/jpeg', 'image/png', 'image/jpg')),
    uiOutput("seleccionred"),
    # uiOutput("radiobuton"),
    uiOutput("botoncargarpeso")
  ), 
  
  mainPanel(
    tabsetPanel(type='pills',
                tabPanel("Painting Classification"
                         ),
                tabPanel("Photograph or Painting Classification",
                         # uiOutput("imagencargada"),
                         plotOutput("imagenc"),
                         # verbatimTextOutput("modelosel"),
                         verbatimTextOutput("resultado"))
                )
    
    
  )
  )
server <- function(input, output, session) {
  imagen <- NULL
  
  obtenerpath <- reactive({
    # browser()
    # if(is.null(input$file$datapath)){return()}
    image <- NULL
    arc <- input$file
    if(is.null(input$file)) {

      return()}
    else
      # { print(gsub("\\\\", "/", arc$datapath))
    {
        imagen <<- load.image(arc$datapath)
        return(imagen)
      }
  })
  
  
  # output$radiobutton <- renderUI({
  #   if (is.null(obtenerpath())) {return()}
  #   actionButton("tipoexp", "Choose the experiment", c("Painting Classification", "Photo vs Painting"))
  # 
  # })
  
  output$seleccionred <- renderUI({
    if (is.null(obtenerpath())) {return()}
    selectInput("red", "Choose the model to use", c("Simple Network", "VGG-16 Network", "ResNet-50 Network"))
  })
  
  output$botoncargarpeso <- renderUI({
    if (is.null(obtenerpath())) {return()}
    actionButton("botoncarga", "Classify the image!")
  })
  
  observeEvent(input$botoncarga,
              { 
               
               modelo <- loadModel()
               if(!dir.exists("1"))
               {
                 dir.create("1")
                 dir.create("1/2")
               }
               # browser()
               # img = obtenerpath()
               # 
               # # img = load.image(path)
               # 
               # print(typeof(img))
               img <- imager::resize(imagen, 224,224)
               
               imager::save.image(img, "1/2/picture.jpg")
               
               
               test <- flow_images_from_directory("1",
                                                  target_size=c(224,224),
                                                  batch_size = 32,
                                                  class_mode = 'binary',
                                                  shuffle = FALSE)
               resfinal = NULL
               res <- modelo %>%
                 predict_generator(test, steps = 1)
               
               if (res <= 0.5)
                 resfinal = "Painting"
               
               else resfinal = "Photograph"
               print(res)
               output$resultado <- renderPrint({paste("Classified as:", resfinal, "with the following score:", res)})
              }
             
               
               
               )
  getModel <- reactive({
    if(is.null(input$botoncarga)) {return()}
    input$botoncarga
    isolate({
      if (input$botoncarga == 0) {return()}
      
      modelo <- input$red
      
      
    })
    return(modelo)
  })
  
  loadModel <- reactive({
    if (is.null(getModel())) {return()}
    
    modelo <- getModel()
    redneuronal <- NULL
    if (modelo == "Simple Network")
    {
      ## cargar pesos red simple
     redneuronal <- keras::load_model_hdf5("simple_bin_phvp32Adam.h5")
    }
    else if (modelo == "VGG-16 Network")
    {
      # redneuronal <- keras::load_model_hdf5("pesos vgg")
    }
    else
    {
      # redneuronal <- # keras::load_model_hdf5("pesos resnet")
    }
    
    return(redneuronal)
  })
  

  output$modelosel <- renderPrint({
    getModel()
  })
  
  
  observeEvent(input$file, 
               {
                 output$imagenc <- renderPlot({
                   plot(imagen)
                   
                 })
                 
                 output$resultado <- NULL
               })

  
  
  # output$imagencargada <- renderUI({
  #   img <- obtenerpath()
  #   if(is.null(img)) {
  # 
  #     output$imagen <- NULL
  #     }
  #   # 
  #   # else{
  #   #   
  #   #   output$imagen <- renderPlot({
  #   #     plot(img)
  #   #     
  #   #   })
  #   # } 
  #   })
  
}

shinyApp(ui, server)
