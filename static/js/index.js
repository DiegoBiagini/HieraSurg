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
