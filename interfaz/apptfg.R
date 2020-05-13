library(shiny)
library(keras)
library(reticulate)
library(imager)
library(shinythemes)
library(purrr)
library(R6)
library(gridExtra)


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
                           
                         )), 
                tabPanel("Style Transfer",
                         sidebarPanel(
                           fileInput("filebase", "Load the image to transform", accept = c('image/jpeg', 'image/png', 'image/jpg')),
                           fileInput("filestyle", "Load the style reference", accept = c('image/jpeg', 'image/png', 'image/jpg')),
                           uiOutput("iteraciones"),
                           uiOutput("botonempezartransfer")
                         ),
                         mainPanel(
                          fluidRow(
                            splitLayout(cellWidths = c("50%", "50%"), plotOutput("imgbase"), plotOutput("imgestilo"))
                          ),
                           
                            
                           plotOutput("imgstyle")
                         )
                         )
                )
    
    
  # )
  )
options(shiny.maxRequestSize=30*1024^2)
server <- function(input, output, session) {
  imagenfoto <- NULL
  imagenpaint <- NULL
  imagebase <- NULL
  imagestyle <- NULL
  
  
  #### funciones importadas de neural_style_transfer ####
  total_variation_weight <- 1
  style_weight <- 1
  content_weight <- 0.025
  
  # util function to open, resize and format pictures into appropriate tensors
  preprocess_image <- function(path){

    img <- image_load(path, target_size = c(img_nrows, img_ncols)) %>%
      image_to_array() %>%
      array_reshape(c(1, dim(.)))
    imagenet_preprocess_input(img)
  }

  # util function to convert a tensor into a valid image
  # also turn BGR into RGB.
  deprocess_image <- function(x){
    x <- x[1,,,]
    # Remove zero-center by mean pixel
    x[,,1] <- x[,,1] + 103.939
    x[,,2] <- x[,,2] + 116.779
    x[,,3] <- x[,,3] + 123.68
    # BGR -> RGB
    x <- x[,,c(3,2,1)]
    # clip to interval 0, 255
    x[x > 255] <- 255
    x[x < 0] <- 0
    x[] <- as.integer(x)/255
    x
  }

  # compute gram matrix of image tensor
  gram_matrix <- function(x){

    features <- x %>%
      k_permute_dimensions(pattern = c(3, 1, 2)) %>%
      k_batch_flatten()

    k_dot(features, k_transpose(features))
  }

  # style loss computation
  style_loss <- function(style, combination){
    S <- gram_matrix(style)
    C <- gram_matrix(combination)

    channels <- 3
    size <- img_nrows*img_ncols

    k_sum(k_square(S - C)) / (4 * channels^2  * size^2)
  }

  # mantain the content of the base image
  content_loss <- function(base, combination){
    k_sum(k_square(combination - base))
  }


  # total variation loss, keeps the image locally coherent

  total_variation_loss <- function(x){
    y_ij  <- x[,1:(img_nrows - 1L), 1:(img_ncols - 1L),]
    y_i1j <- x[,2:(img_nrows), 1:(img_ncols - 1L),]
    y_ij1 <- x[,1:(img_nrows - 1L), 2:(img_ncols),]

    a <- k_square(y_ij - y_i1j)
    b <- k_square(y_ij - y_ij1)
    k_sum(k_pow(a + b, 1.25))
  }

  eval_loss_and_grads <- function(image){
    image <- array_reshape(image, c(1, img_nrows, img_ncols, 3))
    outs <- f_outputs(list(image))
    list(
      loss_value = outs[[1]],
      grad_values = array_reshape(outs[[2]], dim = length(outs[[2]]))
    )
  }
  # Loss and gradients evaluator.
  #
  # This Evaluator class makes it possible
  # to compute loss and gradients in one pass
  # while retrieving them via two separate functions,
  # "loss" and "grads". This is done because scipy.optimize
  # requires separate functions for loss and gradients,
  # but computing them separately would be inefficient.
  Evaluator <- R6Class(
    "Evaluator",
    public = list(

      loss_value = NULL,
      grad_values = NULL,

      initialize = function() {
        self$loss_value <- NULL
        self$grad_values <- NULL
      },

      loss = function(x){
        loss_and_grad <- eval_loss_and_grads(x)
        self$loss_value <- loss_and_grad$loss_value
        self$grad_values <- loss_and_grad$grad_values
        self$loss_value
      },

      grads = function(x){
        grad_values <- self$grad_values
        self$loss_value <- NULL
        self$grad_values <- NULL
        grad_values
      }

    )
  )
  
  
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
                                                    shuffle = FALSE
                                                    )
                 resfinal = NULL
                 resphoto <- modelo %>%
                   predict_generator(test, steps = 1)
                 
                 if (resphoto <= 0.5)
                   resfinal = "painting"
                 
                 else resfinal = "photograph"
                 
                 k_clear_session()
                 
                 if (resfinal == "painting")
                 {
                   modelo <- keras::load_model_hdf5("pesos/estilos/vgg.h5")
                   test <- flow_images_from_directory("1",
                                                      target_size=c(224,224),
                                                      batch_size = 32,
                                                      class_mode = 'categorical',
                                                      shuffle = FALSE
                   )
                   
                   res2 <- modelo %>%
                     predict_generator(test, steps = 1)
                   
                   clases <- whichpartrev(res2)
                   
                   clastop5 <- c()
                   
                   for (res in 1:length(clases))
                   {
                     if ((clases[res] - 1) == 0)
                     {
                       resf = "Abstract Expressionism"
                     }
                     else if ((clases[res] - 1)  == 1)
                     {
                       resf = "Art Nouveau (Modern)"
                     }
                     else if ((clases[res] - 1)  == 2)
                     {
                       resf = "Baroque"
                     }
                     else if ((clases[res] - 1)  == 3)
                     {
                       resf = "Cubism"
                     }
                     else if ((clases[res] - 1)  == 4)
                     {
                       resf = "Early Renaissance"
                     }
                     else if ((clases[res] - 1)  == 5)
                     {
                       resf = "Expressionism"
                     }
                     else if ((clases[res] - 1)  == 6)
                     {
                       resf = "Impressionism"
                     }
                     else if ((clases[res] - 1)  == 7)
                     {
                       resf = "Mannerism (Late Renaissance)"
                     }
                     else if ((clases[res] - 1)  == 8)
                     {
                       resf = "Naïve Art (Primitivism)"
                     }
                     else if ((clases[res] - 1) == 9)
                     {
                       resf = "Northern Renaissance"
                     }
                     else if ((clases[res] - 1)  == 10)
                     {
                       resf = "Post Impressionism"
                     }
                     
                     else if ((clases[res] - 1) == 11)
                     {
                       resf = "Realism"
                     }#
                     else if ((clases[res] - 1) == 12)
                     {
                       resf = "Rococo"
                     }
                     else if ((clases[res] - 1) == 13)
                     {
                       resf = "Romanticism"
                     }
                     else if ((clases[res] - 1) == 14)
                     {
                       resf = "Surrealism"
                     }
                     else if ((clases[res] - 1) == 15)
                     {
                       resf = "Symbolism"
                     }
                     
                     
                     clastop5 <- c(clastop5, resf)
                   }
                   
                 }
                 
                 k_clear_session()
                 
                 output$resultadofoto <- renderUI({
                   
                   strf <- HTML(paste("Classified as a <b>", resfinal, "</b> <br> Obtained score <b>", round(resphoto, 4), "</b><br>"))
                   
               
                   if (resfinal == "painting")
                   {
                     str1 <- "Possible styles:<br>"
                     str2 <- "<ul>"
                     
                     for (clase in clastop5)
                     {
                       str2 <- paste(str2, "<li>", clase, "</li>")
                     }
                     str2 <- paste0(str2, "</ul>")
                     
                     HTML(paste0(strf, str1, str2))
                   }
                   else
                   {
                     HTML(strf)
                   }

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
    selectInput("redpaint", "Choose the model to use", c("Simple Network", "VGG-16 Network"))
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
                                  class_mode = 'categorical',
                                  shuffle = FALSE)
                 resfinal = NULL
                 cla <- modelo %>%
                    predict_generator(test, steps = 1)

                 
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
                 
                 
                 
                 k_clear_session()
                 
                 output$resultadopaint <- renderUI({
                    str1 <- "Possible styles:<br>"
                    str2 <- "<ul>"
                    
                    for (clase in clastop5)
                    {
                      str2 <- paste(str2, "<li>", clase, "</li>")
                    }
                    str2 <- paste0(str2, "</ul>")
                    # str3 <- paste0("Style with highest probability score:<br><b>", resfinal)
                    # 
                    strfinal <- paste(str1, str2)
                    
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
      redneuronal <- keras::load_model_hdf5("pesos/estilos/vgg.h5")
    }
    # else
    # {
    #   redneuronal <- keras::load_model_hdf5("pesos/estilos/resnet.h5")
    # }
    
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

  
  
  #### elementos style transfer ####
  
  obtenerpathbase <- reactive({
    #guardamos el path de la foto base

    arc <- input$filebase
    if(is.null(arc)) {
      
      return()}
    else
      # { print(gsub("\\\\", "/", arc$datapath))
    {
      imagebase <<- as.character(arc$datapath)
      
      return(arc$datapath)
    }
  }) 
  
  obtenerpathestilo <- reactive({
    #guardamos el path de la foto estilo
    
    arc <- input$filestyle
    if(is.null(arc)) {
      
      return()}
    else
      # { print(gsub("\\\\", "/", arc$datapath))
    {
      imagestyle <<- as.character(arc$datapath)
      
      return(arc$datapath)
    }
  })
  output$iteraciones <- renderUI({
    if(is.null(obtenerpathbase()) || is.null(obtenerpathestilo()))
    {
      return()
    }
    numericInput("iterations", "Number of iterations: ", value = 10)
  })
  output$botonempezartransfer <- renderUI({
    if(is.null(obtenerpathbase()) || is.null(obtenerpathestilo()))
    {
      return()
    }
    actionButton("botonempezar", "Start style transfer!")
  })
  
  output$imgbase <- renderPlot({
    if (is.null(obtenerpathbase()))
      {
        return()
    }
    foto <- imager::load.image(imagebase)
    plot(foto, main = "Base Image")
  })
  output$imgestilo <- renderPlot({
    if (is.null(obtenerpathestilo()))
    {
      return()
    }
    foto <- imager::load.image(imagestyle)
    plot(foto, main = "Style Reference")
  })
  
  img_nrows <- 0
  img_ncols <- 0
  f_outputs <- NULL
  #### observe event para style transfer ####
  observeEvent(input$botonempezar, 
               {
                 if (input$iterations < 1)
                 {
                   return()
                 }
                 output$imgstyle <- NULL
                 tensorflow::tf$compat$v1$disable_eager_execution()
                 iterations <- input$iterations
                 print(iterations)
                 img <- image_load(imagebase)
                 
                 width <- img$size[[1]]
                 height <- img$size[[2]]
                 img_nrows <<- 400
                 img_ncols <<- as.integer(width * img_nrows / height)
                 
                 # tensor representation of the images
                
                 base_image <- k_variable(preprocess_image(imagebase))
                 style_reference_image <- k_variable(preprocess_image(imagestyle))
                 
                 # combination container
                 
                 combination_image <- k_placeholder(c(1, img_nrows, img_ncols, 3))
                 
                 # combine the 3 images into a single keras tensor
                 # combine the 3 images into a single Keras tensor
                 input_tensor <- k_concatenate(list(base_image, style_reference_image, 
                                                    combination_image), axis = 1)
                 
                 # model creation
                 
                 model <- application_vgg16(input_tensor = input_tensor, weights = "pesos_vgg_impbar.h5", 
                                            include_top = FALSE)
                 
                 nms <- map_chr(model$layers, ~.x$name)
                 output_dict <- map(model$layers, ~.x$output) %>% set_names(nms)
                 
                 # combine these loss functions into a single scalar
                 loss <- k_variable(0.0)
                 layer_features <- output_dict$block4_conv2
                 base_image_features <- layer_features[1,,,]
                 combination_features <- layer_features[3,,,]
                 
                 loss <- loss + content_weight*content_loss(base_image_features, 
                                                            combination_features)
                 
                 # which layers to use for features
                 feature_layers = c('block1_conv1', 'block2_conv1',
                                    'block3_conv1', 'block4_conv1',
                                    'block5_conv1')
                 
                 
                 for(layer_name in feature_layers){
                   layer_features <- output_dict[[layer_name]]
                   style_reference_features <- layer_features[2,,,]
                   combination_features <- layer_features[3,,,]
                   sl <- style_loss(style_reference_features, combination_features)
                   loss <- loss + ((style_weight / length(feature_layers)) * sl)
                 }
                 
                 loss <- loss + (total_variation_weight * total_variation_loss(combination_image))
                 
                 
                 grads <- k_gradients(loss, combination_image)[[1]]
                 
                 f_outputs <<- k_function(list(combination_image), list(loss, grads))
                 

                 
                 evaluator <- Evaluator$new()
                 
                 dms <- c(1, img_nrows, img_ncols, 3)
                 x <- array(data = runif(prod(dms), min = 0, max = 255) - 128, dim = dms)
                 
                 # Run optimization (L-BFGS) over the pixels of the generated image
                 # so as to minimize the loss
                 im <- NULL
                 imageaux <- NULL
                # TODO: arreglar tiempo
                 withProgress(message = "Generating combined picture", value=0, style = "notification",{
                   for(i in 1:iterations){
                     incProgress(0.1, detail = paste0("Iteration: ", i,"/", iterations))
                     # Run L-BFGS
                     opt <- optim(
                       array_reshape(x, dim = length(x)), fn = evaluator$loss, gr = evaluator$grads, 
                       method = "L-BFGS-B",
                       control = list(maxit = 15)
                     )
                     
                     # Print loss value
                     # print(opt$value)
                     
                     # decode the image
                     imageaux <- x <- opt$par
                     
                     ## TODO: MIRAR FALLOS A PARTIR DE AQUÍ: cannot reshape array of size 1 into shape (1, 400, 319, 3) EN LA SIGUIENTE LINEA
                     imageaux <- array_reshape(imageaux, dms)
                     
                     # png(paste0("styleprueba/", i, ".png"))
                     # im <- deprocess_image(image)
                     # plot(as.raster(im))
                     # dev.off()
                     
                   }
                   
                 } )
                 
                 im <- deprocess_image(imageaux)
                 k_clear_session()
                 output$imgstyle <- renderPlot({
                   plot(as.raster(im), main = "Result")
                 })
                 })
 
}
shinyApp(ui, server)
