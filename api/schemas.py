from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    has_title: int = Field(..., ge=0)
    has_input: int = Field(..., ge=0)
    has_button: int = Field(..., ge=0)
    has_image: int = Field(..., ge=0)
    has_submit: int = Field(..., ge=0)
    has_link: int = Field(..., ge=0)
    has_password: int = Field(..., ge=0)
    has_email_input: int = Field(..., ge=0)
    has_hidden_element: int = Field(..., ge=0)
    has_audio: int = Field(..., ge=0)
    has_video: int = Field(..., ge=0)

    number_of_inputs: int = Field(..., ge=0)
    number_of_buttons: int = Field(..., ge=0)
    number_of_images: int = Field(..., ge=0)
    number_of_option: int = Field(..., ge=0)
    number_of_list: int = Field(..., ge=0)
    number_of_th: int = Field(..., ge=0)
    number_of_tr: int = Field(..., ge=0)
    number_of_href: int = Field(..., ge=0)
    number_of_paragraph: int = Field(..., ge=0)
    number_of_script: int = Field(..., ge=0)

    length_of_title: int = Field(..., ge=0)

    has_h1: int = Field(..., ge=0)
    has_h2: int = Field(..., ge=0)
    has_h3: int = Field(..., ge=0)

    length_of_text: int = Field(..., ge=0)

    number_of_clickable_button: int = Field(..., ge=0)
    number_of_a: int = Field(..., ge=0)
    number_of_img: int = Field(..., ge=0)
    number_of_div: int = Field(..., ge=0)
    number_of_figure: int = Field(..., ge=0)

    has_footer: int = Field(..., ge=0)
    has_form: int = Field(..., ge=0)
    has_text_area: int = Field(..., ge=0)
    has_iframe: int = Field(..., ge=0)
    has_text_input: int = Field(..., ge=0)

    number_of_meta: int = Field(..., ge=0)

    has_nav: int = Field(..., ge=0)
    has_object: int = Field(..., ge=0)
    has_picture: int = Field(..., ge=0)

    number_of_sources: int = Field(..., ge=0)
    number_of_span: int = Field(..., ge=0)
    number_of_table: int = Field(..., ge=0)