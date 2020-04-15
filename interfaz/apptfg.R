library(shiny)
library(keras)
library(reticulate)
library(imager)
ui <- fluidPage(
  
  titlePanel("Painting Classification", windowTitle = "TFG"),
  # sidebarPanel(
  #   fileInput("file", "Load the image", accept = c('image/jpeg', 'image/png', 'image/jpg')),
  #   uiOutput("seleccionred"),
  #   # uiOutput("radiobuton"),
  #   uiOutput("botoncargarpeso")
  # ), 
  
  # mainPanel(
    tabsetPanel(type='pills',
                tabPanel("Style Classification",
                         sidebarPanel(
                           fileInput("filepaint", "Load the image", accept = c('image/jpeg', 'image/png', 'image/jpg')),
                           uiOutput("seleccionredpaint"),
                           uiOutput("botoncargarpesopaint")
                         ),
                         mainPanel(
                           plotOutput("imagencpaint"),
                           uiOutput("resultadopaint")
                         )
                         ),
                tabPanel("Photograph or Painting Classification",
                         sidebarPanel(
                           fileInput("filefoto", "Load the image", accept = c('image/jpeg', 'image/png', 'image/jpg')),
                           uiOutput("seleccionredfoto"),
                           # uiOutput("radiobuton"),
                           uiOutput("botoncargarpesofoto")
                         ),
                         
                         mainPanel(
                           # uiOutput("imagencargada"),
                           plotOutput("imagencfoto"),
                           # verbatimTextOutput("modelosel"),
                           htmlOutput("resultadofoto")
                         ))
                )
    
    
  # )
  )
options(shiny.maxRequestSize=30*1024^2)
server <- function(input, output, session) {
  imagenfoto <- NULL
  imagenpaint <- NULL
  

  #### elementos experimentos photo ####
  obtenerpathfoto <- reactive({
    # browser()
    # if(is.null(input$file$datapath)){return()}
    image <- NULL
    arc <- input$filefoto
    if(is.null(arc)) {
      
      return()}
    else
      # { print(gsub("\\\\", "/", arc$datapath))
    {
      imagenfoto <<- load.image(arc$datapath)
      return(imagenfoto)
    }
  })
  
  output$seleccionredfoto <- renderUI({
    if (is.null(obtenerpathfoto())) {return()}
    selectInput("redfoto", "Choose the model to use", c("Simple Network", "VGG-16 Network", "ResNet-50 Network"))
  })
  
  output$botoncargarpesofoto <- renderUI({
    if (is.null(obtenerpathfoto())) {return()}
    actionButton("botoncargafoto", "Classify the image!")
  })
  
  observeEvent(input$botoncargafoto,
               { 
                 
                 modelo <- loadModelfoto()
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
                 img <- imager::resize(imagenfoto, 224,224)
                 
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
                   resfinal = "painting"
                 
                 else resfinal = "photograph"

                 
                 output$resultadofoto <- renderUI({
                   
                   strfinal <- HTML(paste("<font size=+2>Classified as a <b>", resfinal, "</b> <br> Obtained score <b>", round(res, 4), "</b></font>"))
                   
                   
                   strfinal
                   })
               }
               
               
               
  )
  getModelfoto <- reactive({
    if(is.null(input$botoncargafoto)) {return()}
    input$botoncargafoto
    isolate({
      if (input$botoncargafoto == 0) {return()}
      
      modelo <- input$redfoto
      
      
    })
    return(modelo)
  })
  
  loadModelfoto <- reactive({
    if (is.null(getModelfoto())) {return()}
    
    modelo <- getModelfoto()
    redneuronal <- NULL
    if (modelo == "Simple Network")
    {
      ## cargar pesos red simple
      redneuronal <- keras::load_model_hdf5("pesos/fotos/simple_bin_phvp2.h5")
    }
    else if (modelo == "VGG-16 Network")
    {
      redneuronal <- keras::load_model_hdf5("pesos/fotos/vgg16_bin_SGD_32_phpa.h5")
    }
    else
    {
      # redneuronal <- # keras::load_model_hdf5("pesos resnet")
    }
    
    return(redneuronal)
  })
  
  
  # output$modelosel <- renderPrint({
  #   getModel()
  # })
  # 
  
  observeEvent(input$filefoto, 
               {
                 output$imagencfoto <- renderPlot({
                   plot(imagenfoto)
                   
                 })
                 
                 output$resultadofoto <- NULL
               })
  
  observeEvent(input$redfoto, 
               {
                 output$resultadofoto <- NULL
               })
  #### elementos experimentos estilos ####
  
 
  obtenerpathpaint <- reactive({
    # browser()
    # if(is.null(input$file$datapath)){return()}
    image <- NULL
    arc <- input$filepaint
    if(is.null(arc)) {
      
      return()}
    else
      # { print(gsub("\\\\", "/", arc$datapath))
    {
      imagenpaint <<- load.image(arc$datapath)
      return(imagenpaint)
    }
  })
  
  output$seleccionredpaint <- renderUI({
    if (is.null(obtenerpathpaint())) {return()}
    selectInput("redpaint", "Choose the model to use", c("Simple Network", "VGG-16 Network", "ResNet-50 Network"))
  })
  
  output$botoncargarpesopaint <- renderUI({
    if (is.null(obtenerpathpaint())) {return()}
    actionButton("botoncargapaint", "Classify the painting!")
  })
  
  observeEvent(input$botoncargapaint,
               { 
                 
                 # modelo <- loadModelpaint()
                 # if(!dir.exists("1"))
                 # {
                 #   dir.create("1")
                 #   dir.create("1/2")
                 # }
                 # # browser()
                 # # img = obtenerpath()
                 # # 
                 # # # img = load.image(path)
                 # # 
                 # # print(typeof(img))
                 # img <- imager::resize(imagenpaint, 224,224)
                 # 
                 # imager::save.image(img, "1/2/picture.jpg")
                 # 
                 # 
                 # test <- flow_images_from_directory("1",
                 #                                    target_size=c(224,224),
                 #                                    batch_size = 32,
                 #                                    class_mode = 'binary',
                 #                                    shuffle = FALSE)
                 # resfinal = NULL
                 # res <- modelo %>%
                 #   predict_generator(test, steps = 1)
                 # 
                 # # este if cambiarlo dependiendo del resultado obtenido, 16 clases, el valor irá de 0 a 15
                 # # if (res <= 0.5)
                 # #   resfinal = "painting"
                 # # 
                 # # else resfinal = "photograph"
                
                 output$resultadopaint <- renderUI({
                   
                   # strfinal <- HTML(paste("<font size=+2>Classified as a <b>", resfinal, "</b> <br> Obtained score <b>", round(res, 4), "</b></font>"))
                   strfinal <- HTML(paste("<font size=+2> Work in progress!! UwU</font>"))
                   
                   strfinal
                 })
               }
               
               
               
  )
  getModelpaint <- reactive({
    if(is.null(input$botoncargapaint)) {return()}
    input$botoncargapaint
    isolate({
      if (input$botoncargapaint == 0) {return()}
      
      modelo <- input$redpaint
      
      
    })
    return(modelo)
  })
  
  loadModelpaint <- reactive({
    if (is.null(getModelpaint())) {return()}
    
    modelo <- getModelpaint()
    redneuronal <- NULL
    if (modelo == "Simple Network")
    {
      ## cargar pesos red simple
      # redneuronal <- keras::load_model_hdf5("pesos/estilos/simple")
    }
    else if (modelo == "VGG-16 Network")
    {
      # redneuronal <- keras::load_model_hdf5("pesos/estilos/vgg")
    }
    else
    {
      # redneuronal <- keras::load_model_hdf5("pesos/estilos/resnet")
    }
    
    return(redneuronal)
  })
  

  observeEvent(input$filepaint, 
               {
                 output$imagencpaint <- renderPlot({
                   plot(imagenpaint)
                   
                 })
                 
                 output$resultadopaint <- NULL
               })
  
  observeEvent(input$redpaint, 
               {
                 output$resultadopaint <- NULL
               })

  
  
}

shinyApp(ui, server)
