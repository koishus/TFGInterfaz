library(shiny)
library(keras)
library(reticulate)
library(imager)
library(shinythemes)

ui <- fluidPage(
  theme = shinythemes::shinytheme("yeti"),
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
  
  whichpartrev <- function(x, n=5) {
    which(x >= -sort(-x, partial=n)[n])
  }
  
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
                 
                 k_clear_session()
                 output$resultadofoto <- renderUI({
                   
                   HTML(paste("<font size=+2>Classified as a <b>", resfinal, "</b> <br> Obtained score <b>", round(res, 4), "</b></font>"))
                 
                   
                   
                   
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
      redneuronal <- keras::load_model_hdf5("pesos/fotos/resnet_phvp.h5")
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

    image <- NULL
    arc <- input$filepaint
    if(is.null(arc)) {
      
      return()}
    else
     
    {
      imagenpaint <<- load.image(arc$datapath)
      if(!dir.exists("1"))
      {
        dir.create("1")
        dir.create("1/2")
      }

      img <- imager::resize(imagenpaint, 224,224)

      imager::save.image(img, "1/2/picture.jpg")
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
                 
               modelo <- loadModelpaint()


                 test <- flow_images_from_directory("1",
                                target_size=c(224,224),
                                   batch_size = 32,
                                  class_mode = 'binary',
                                  shuffle = FALSE)
                 resfinal = NULL
                 cla <- modelo %>%
                    predict_generator(test, steps = 1)
                 #
                 # # este if cambiarlo dependiendo del resultado obtenido, 16 clases, el valor irá de 0 a 15
                 # # if (res <= 0.5)
                 # #   resfinal = "painting"
                 # #
                 # # else resfinal = "photograph"
                 
                 clases <- whichpartrev(cla)
                 clastop5 <- c()
                 
                 for (res in 1:length(clases))
                 {
                   if ((clases[res] - 1) == 0)
                   {
                     resfinal = "Abstract Expressionism"
                   }
                   else if ((clases[res] - 1)  == 1)
                   {
                     resfinal = "Art Nouveau (Modern)"
                   }
                   else if ((clases[res] - 1)  == 2)
                   {
                     resfinal = "Baroque"
                   }
                   else if ((clases[res] - 1)  == 3)
                   {
                     resfinal = "Cubism"
                   }
                   else if ((clases[res] - 1)  == 4)
                   {
                     resfinal = "Early Renaissance"
                   }
                   else if ((clases[res] - 1)  == 5)
                   {
                     resfinal = "Expressionism"
                   }
                   else if ((clases[res] - 1)  == 6)
                   {
                     resfinal = "Impressionism"
                   }
                   else if ((clases[res] - 1)  == 7)
                   {
                     resfinal = "Mannerism (Late Renaissance)"
                   }
                   else if ((clases[res] - 1)  == 8)
                   {
                     resfinal = "Naïve Art (Primitivism)"
                   }
                   else if ((clases[res] - 1) == 9)
                   {
                     resfinal = "Northern Renaissance"
                   }
                   else if ((clases[res] - 1)  == 10)
                   {
                     resfinal = "Post Impressionism"
                   }
                   
                   else if ((clases[res] - 1) == 11)
                   {
                     resfinal = "Realism"
                   }#
                   else if ((clases[res] - 1) == 12)
                   {
                     resfinal = "Rococo"
                   }
                   else if ((clases[res] - 1) == 13)
                   {
                     resfinal = "Romanticism"
                   }
                   else if ((clases[res] - 1) == 14)
                   {
                     resfinal = "Surrealism"
                   }
                   else if ((clases[res] - 1) == 15)
                   {
                     resfinal = "Symbolism"
                   }
                   
                   clastop5 <- c(clastop5, resfinal)
                 }
                 
                 res <- which.max(cla)-1
                 
                 resfinal <- NULL
                 if (res == 0)
                 {
                   resfinal = "Abstract Expressionism"
                 }
                 else if (res == 1)
                 {
                   resfinal = "Art Nouveau (Modern)"
                 }
                 else if (res == 2)
                 {
                   resfinal = "Baroque"
                 }
                 else if (res == 3)
                 {
                   resfinal = "Cubism"
                 }
                 else if (res == 4)
                 {
                   resfinal = "Early Renaissance"
                 }
                 else if (res == 5)
                 {
                   resfinal = "Expressionism"
                 }
                 else if (res == 6)
                 {
                   resfinal = "Impressionism"
                 }
                 else if (res == 7)
                 {
                   resfinal = "Mannerism (Late Renaissance)"
                 }
                 else if (res == 8)
                 {
                   resfinal = "Naïve Art (Primitivism)"
                 }
                 else if (res == 9)
                 {
                   resfinal = "Northern Renaissance"
                 }
                 else if (res == 10)
                 {
                   resfinal = "Post Impressionism"
                 }
                 
                 else if (res == 11)
                 {
                   resfinal = "Realism"
                 }#
                 else if (res == 12)
                 {
                   resfinal = "Rococo"
                 }
                 else if (res == 13)
                 {
                   resfinal = "Romanticism"
                 }
                 else if (res == 14)
                 {
                   resfinal = "Surrealism"
                 }
                 else if (res == 15)
                 {
                   resfinal = "Symbolism"
                 }
                 
                 
                 k_clear_session()
                 
                 output$resultadopaint <- renderUI({
                    str1 <- "Possible styles:<br>"
                    str2 <- "<ul>"
                    
                    for (clase in clastop5)
                    {
                      str2 <- paste(str2, "<li>", clase, "</li>")
                    }
                    str2 <- paste0(str2, "</ul>")
                    str3 <- paste0("Style with highest probability score:<br><b>", resfinal)
                    
                    strfinal <- paste(str1, str2, str3)
                    
                    HTML(strfinal)
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
      redneuronal <- keras::load_model_hdf5("pesos/estilos/simplemulti.h5")
    }
    else if (modelo == "VGG-16 Network")
    {
      # redneuronal <- keras::load_model_hdf5("pesos/estilos/vgg")
    }
    else
    {
      redneuronal <- keras::load_model_hdf5("pesos/estilos/resnet.h5")
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
