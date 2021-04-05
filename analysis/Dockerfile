FROM rocker/shiny:3.6.3

RUN apt-get update && apt-get install libcurl4-openssl-dev -y && apt-get install libcurl4-gnutls-dev -y &&\
  mkdir -p /var/lib/shiny-server/bookmarks/shiny

# Download and install libraries
RUN R -e "install.packages(c('shiny', 'ggplot2', 'ggbupbr', 'FactoMineR', 'plotly'))"
RUN R -e "install.packages('devtools', dependencies = TRUE)"

# Copy the app to the image 
COPY shiny_app /srv/shiny-server/

# Make all app files readable
RUN chmod -R 755 /srv/shiny-server/

EXPOSE 3838
CMD ["/usr/bin/shiny-server.sh"
