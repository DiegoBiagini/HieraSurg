window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousel_images = bulmaCarousel.attach('#images-carousel', options);


	var options = {
		infinite: false,
		autoplay: false,
	}
	var carousel_videos = bulmaCarousel.attach('#videos-carousel', options);
	
    bulmaSlider.attach();

})


document
  .querySelector('#videos-carousel')
  .addEventListener('carousel:show', function(e){
  
    // e.detail.index is the zero-based index of the slide just shown
    var idx       = e.detail.index;
    var slides    = this.querySelectorAll('.carousel-item');
    var active    = slides[idx];
    var video     = active.querySelector('video');

    // if there is a video, play it and when it's done, go to next slide
    if (video){
      // rewind / start from 0
      video.currentTime = 0;
      video.play();
      
      // when the video ends, advance the carousel
      video.onended = function(){
        carousel.next();
      };
    }
  });